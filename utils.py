# utils.py
# Native llama.cpp server adapter for KLL535/ComfyUI_Simple_Qwen3-VL-gguf.
# Exposes a small llama-cpp-python-compatible Llama surface while running
# a local upstream llama-server process.

from __future__ import annotations

import atexit
import json
import os
import base64
import mimetypes
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ----------------------------------------------------------------------
# Windows child-process lifetime
# ----------------------------------------------------------------------
# llama-server keeps VRAM allocated for as long as it runs. Python-level
# cleanup (close/__del__/atexit) is skipped when the parent is force-killed:
# taskkill /F, console window close, Task Manager, os._exit, native crash.
# A job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is the only mechanism
# that survives those paths: the kernel closes the dead parent's handles, and
# the job kills every member process once its last handle is gone.

_JOB_HANDLE: Any = None
_JOB_UNAVAILABLE = False


def _win_job_handle() -> Any:
    """Create the process-wide kill-on-close job object (once).

    Raises on the first failure, then latches and returns None afterwards.
    """
    global _JOB_HANDLE, _JOB_UNAVAILABLE

    if _JOB_HANDLE is not None or _JOB_UNAVAILABLE:
        return _JOB_HANDLE

    import ctypes
    from ctypes import wintypes

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryLimit", ctypes.c_size_t),
            ("PeakJobMemoryLimit", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        _JOB_UNAVAILABLE = True
        raise ctypes.WinError(ctypes.get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        handle,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        err = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        _JOB_UNAVAILABLE = True
        raise ctypes.WinError(err)

    # Kept open for the lifetime of this process on purpose: closing it here
    # would immediately kill the assigned llama-server.
    _JOB_HANDLE = handle
    return _JOB_HANDLE


def _win_assign_to_job(proc: subprocess.Popen) -> None:
    """Bind proc to the kill-on-close job object. No-op outside Windows."""
    if os.name != "nt":
        return

    import ctypes
    from ctypes import wintypes

    job = _win_job_handle()
    if not job:
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    # Popen's handle comes from CreateProcess with full access, so it already
    # covers PROCESS_SET_QUOTA | PROCESS_TERMINATE.
    h_proc = wintypes.HANDLE(int(proc._handle))
    if not kernel32.AssignProcessToJobObject(job, h_proc):
        raise ctypes.WinError(ctypes.get_last_error())


_REQUEST_TIMEOUT = 3600.0  # matches llama-server's own --timeout default


class Llama:
    """Compatibility adapter around a local upstream llama-server.

    Only the Llama surface used by qwen3vl_run.py is implemented:
      - create_chat_completion(...)
      - create_completion(...)
      - _ctx.memory_clear(...) / close()
      - chat_handler / n_tokens / is_hybrid / _hybrid_cache_mgr

    The adapter intentionally leaves model/prompt preset logic in qwen3vl_run.py.
    Native-server-only options arrive through the `_native_config` dict that
    qwen3vl_run.py injects, never as explicit keyword arguments.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        swa_full: bool = False,
        verbose: bool = False,
        n_threads: int = 8,
        n_gpu_layers: Any = -1,
        split_mode: Any = 0,
        main_gpu: int = 0,
        ctx_checkpoints: int = 0,
        tensor_split: Optional[Iterable[Any]] = None,
        type_k: Any = None,
        type_v: Any = None,
        n_cpu_moe: Optional[int] = None,
        cpu_moe: Optional[bool] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        n_keep: Optional[int] = None,
        flash_attn_type: Any = None,
        offload_kqv: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        native_config = kwargs.pop("_native_config", None)
        if not isinstance(native_config, dict):
            native_config = {}

        mmproj_path = native_config.get("mmproj_path")
        image_min_tokens = native_config.get("image_min_tokens")
        image_max_tokens = native_config.get("image_max_tokens")
        llama_server_port = native_config.get("llama_server_port")
        llama_server_start_timeout = native_config.get("llama_server_start_timeout", 180.0)
        llama_server_extra_args = native_config.get("llama_server_extra_args")
        self._enable_thinking = (
            bool(native_config["enable_thinking"])
            if "enable_thinking" in native_config
            else None
        )

        self.model_path = str(Path(model_path).expanduser())
        self.chat_handler = None  # llama-server owns the chat template
        self.verbose = bool(verbose)
        self.n_tokens = 0
        self.is_hybrid = False
        self._hybrid_cache_mgr = None
        self._ctx = self
        self._closed = False
        self._force_no_cache_next = False

        self._exe = self._resolve_server_executable(native_config.get("llama_server_path"))
        self._host = "127.0.0.1"
        self._port = int(llama_server_port) if llama_server_port else self._find_free_port()
        self._base_url = f"http://{self._host}:{self._port}"

        cmd: List[str] = [
            self._exe,
            "--host", self._host,
            "--port", str(self._port),
            "--model", self.model_path,
            "--ctx-size", str(int(n_ctx)),
            "--batch-size", str(int(n_batch)),
            "--ubatch-size", str(int(n_ubatch)),
            "--threads", str(int(n_threads)),
            "--n-gpu-layers", self._gpu_layers_value(n_gpu_layers),
            "--split-mode", self._split_mode_value(split_mode),
            "--main-gpu", str(int(main_gpu)),
        ]

        # 0 means "unset" in this project, so leave llama-server's own default.
        if ctx_checkpoints is not None and int(ctx_checkpoints) > 0:
            cmd += ["--ctx-checkpoints", str(int(ctx_checkpoints))]

        if swa_full:
            cmd.append("--swa-full")

        if tensor_split:
            cmd += ["--tensor-split", ",".join(str(x) for x in tensor_split)]

        cache_k = self._cache_type_value(type_k)
        cache_v = self._cache_type_value(type_v)
        if cache_k:
            cmd += ["--cache-type-k", cache_k]
        if cache_v:
            cmd += ["--cache-type-v", cache_v]

        if cpu_moe:
            cmd.append("--cpu-moe")
        elif n_cpu_moe is not None and int(n_cpu_moe) > 0:
            cmd += ["--n-cpu-moe", str(int(n_cpu_moe))]

        # --mmap/--no-mmap/--mlock are deprecated in favour of a single
        # --load-mode value. Both flags off is this project's default, which
        # would translate to "none" and needlessly disable mmap, so leave
        # llama-server on its own "auto" instead.
        load_mode = {
            (True, True): "mmap+mlock",
            (True, False): "mmap",
            (False, True): "mlock",
        }.get((bool(use_mmap), bool(use_mlock)))
        if load_mode:
            cmd += ["--load-mode", load_mode]

        if n_keep is not None:
            cmd += ["--keep", str(int(n_keep))]

        fa = self._flash_attn_value(flash_attn_type)
        if fa:
            cmd += ["--flash-attn", fa]

        if offload_kqv is not None:
            cmd.append("--kv-offload" if offload_kqv else "--no-kv-offload")

        if mmproj_path:
            cmd += ["--mmproj", str(mmproj_path)]
            # 0 means "unset"; llama-server reads its own default from the model.
            if image_min_tokens:
                cmd += ["--image-min-tokens", str(int(image_min_tokens))]
            if image_max_tokens:
                cmd += ["--image-max-tokens", str(int(image_max_tokens))]

        # Only escape hatch for llama.cpp CLI flags: configurator.py has no
        # widgets for them. Prefer a JSON list; strings are also accepted.
        cmd.extend(self._normalize_extra_args(llama_server_extra_args))

        if self.verbose:
            print("[native-llama.cpp] " + subprocess.list2cmdline(cmd), file=sys.stderr)

        popen_kwargs: Dict[str, Any] = {
            "cwd": str(Path(self._exe).parent),
        }
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creation:
                popen_kwargs["creationflags"] = creation
        else:
            popen_kwargs["start_new_session"] = True

        self._process = subprocess.Popen(cmd, **popen_kwargs)

        # Assign immediately: any work in between widens the window in which a
        # parent crash would leave llama-server unmanaged.
        try:
            _win_assign_to_job(self._process)
        except Exception as exc:
            print(
                f"[native-llama.cpp] job object unavailable, llama-server may "
                f"outlive this process: {exc}",
                file=sys.stderr,
            )

        atexit.register(self.close)
        try:
            self._wait_until_ready(float(llama_server_start_timeout))
        except Exception:
            # Otherwise the server keeps running (and holding VRAM) while the
            # half-built object stays alive through the atexit registration.
            self.close()
            raise

    # ------------------------------------------------------------------
    # llama-cpp-python-compatible public surface
    # ------------------------------------------------------------------

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Any = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
        response_format: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages, "stream": False}
        self._put(payload, "max_tokens", max_tokens)
        self._put(payload, "temperature", temperature)
        self._put(payload, "top_p", top_p)
        self._put(payload, "top_k", top_k)
        self._put(payload, "min_p", min_p)
        self._put(payload, "seed", seed)
        self._put(payload, "stop", stop)
        self._put(payload, "repeat_penalty", repeat_penalty)
        self._put(payload, "frequency_penalty", frequency_penalty)
        self._put(payload, "presence_penalty", presence_penalty)
        self._put(payload, "response_format", response_format)
        self._put(payload, "tools", tools)
        self._put(payload, "tool_choice", tool_choice)
        self._merge_request_kwargs(payload, kwargs)

        if self._enable_thinking is not None:
            chat_template_kwargs = payload.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            chat_template_kwargs.setdefault("enable_thinking", self._enable_thinking)
            payload["chat_template_kwargs"] = chat_template_kwargs

        self._apply_cache_policy(payload)

        result = self._post_json("/v1/chat/completions", payload)
        self._ensure_usage(result)
        return result

    def create_completion(
        self,
        prompt: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Any = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": prompt, "stream": False}
        self._put(payload, "max_tokens", max_tokens)
        self._put(payload, "temperature", temperature)
        self._put(payload, "top_p", top_p)
        self._put(payload, "top_k", top_k)
        self._put(payload, "min_p", min_p)
        self._put(payload, "seed", seed)
        self._put(payload, "stop", stop)
        self._put(payload, "repeat_penalty", repeat_penalty)
        self._put(payload, "frequency_penalty", frequency_penalty)
        self._put(payload, "presence_penalty", presence_penalty)
        self._merge_request_kwargs(payload, kwargs)
        self._apply_cache_policy(payload)

        result = self._post_json("/v1/completions", payload)
        self._ensure_usage(result)
        return result

    def memory_clear(self, *_args: Any, **_kwargs: Any) -> None:
        # llama-server owns the slots. The next request is explicitly marked
        # cache_prompt=false, which prevents reusing the previous prompt state.
        self._force_no_cache_next = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        proc = getattr(self, "_process", None)
        if proc is None or proc.poll() is not None:
            return

        # On Windows terminate() is TerminateProcess, and llama-server spawns
        # no children of its own, so there is nothing to walk.
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Server/process helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_server_executable(value: Optional[str]) -> str:
        candidates: List[Path] = []
        if value:
            p = Path(value).expanduser()
            if p.is_dir():
                candidates += [p / "llama-server.exe", p / "llama-server"]
            else:
                candidates.append(p)

        for name in ("llama-server.exe", "llama-server"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

        for p in candidates:
            if p.is_file():
                return str(p.resolve())

        raise FileNotFoundError(
            "llama-server executable not found. Set config['llama_server_path'] "
            "to llama-server.exe or its directory."
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + max(1.0, timeout)
        url = self._base_url + "/health"
        last_error = ""
        while time.monotonic() < deadline:
            rc = self._process.poll()
            if rc is not None:
                raise RuntimeError(f"llama-server exited during startup (code={rc})")
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.HTTPError as exc:
                # 503 means the model is still loading.
                last_error = f"HTTP {exc.code}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise TimeoutError(
            f"llama-server did not become ready within {timeout:g}s"
            + (f"; last error: {last_error}" if last_error else "")
        )

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # KLL535/ComfyUI may pass Load Image / temp media as file:// URLs.
        # llama-server blocks arbitrary file:// access unless --media-path is set,
        # so inline local media into the request as data: URLs instead.
        payload = self._inline_local_media(payload)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + endpoint,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server {endpoint} failed: HTTP {exc.code}: {body}"
            ) from exc
        except Exception as exc:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}: {exc}"
                ) from exc
            raise

        try:
            result = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from llama-server {endpoint}: {raw[:1000]}") from exc

        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"llama-server error: {result['error']}")
        return result

    def _inline_local_media(self, value: Any) -> Any:
        """Recursively convert file:// URL values to base64 data: URLs."""
        if isinstance(value, list):
            return [self._inline_local_media(v) for v in value]

        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if key == "url" and isinstance(item, str) and item.lower().startswith("file://"):
                    out[key] = self._file_url_to_data_url(item)
                else:
                    out[key] = self._inline_local_media(item)
            return out

        return value

    def _file_url_to_data_url(self, file_url: str) -> str:
        parsed = urllib.parse.urlparse(file_url)
        local_path = urllib.request.url2pathname(parsed.path)

        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            local_path = f"//{parsed.netloc}{local_path}"

        if os.name == "nt" and len(local_path) >= 3 and local_path[0] == "/" and local_path[2] == ":":
            local_path = local_path[1:]

        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local media file not found: {path}")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "application/octet-stream"

        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")

        if self.verbose:
            print(
                f"[native-llama.cpp:MEDIA] inlined {path.name} "
                f"({len(raw)} bytes, {mime})",
                file=sys.stderr,
            )

        return f"data:{mime};base64,{encoded}"

    # ------------------------------------------------------------------
    # Parameter translation
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_layers_value(value: Any) -> str:
        iv = int(value)
        return "all" if iv < 0 else str(iv)

    @staticmethod
    def _split_mode_value(value: Any) -> str:
        return {0: "none", 1: "layer", 2: "row", 3: "tensor"}.get(int(value), "none")

    @staticmethod
    def _cache_type_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        # configurator.py offers the full ggml_type enum; anything llama-server
        # does not accept for --cache-type-k/v maps to None and is left unset.
        return {
            0: "f32",
            1: "f16",
            2: "q4_0",
            3: "q4_1",
            6: "q5_0",
            7: "q5_1",
            8: "q8_0",
            20: "iq4_nl",
            30: "bf16",
        }.get(int(value))

    @staticmethod
    def _flash_attn_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        # configurator.py FLASH_ATTN_TYPES: -1=AUTO, 0=DISABLED, 1=ENABLED
        return {-1: "auto", 0: "off", 1: "on"}.get(int(value))

    @staticmethod
    def _normalize_extra_args(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
        if isinstance(value, str):
            return shlex.split(value, posix=(os.name != "nt"))
        raise TypeError("llama_server_extra_args must be a string or list")

    @staticmethod
    def _put(payload: Dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            payload[key] = value

    def _merge_request_kwargs(self, payload: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        # llama-server's OpenAI endpoints accept many llama.cpp-specific sampler
        # fields in addition to the OpenAI fields, so preserve extra_completion_*.
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

    def _apply_cache_policy(self, payload: Dict[str, Any]) -> None:
        if self._force_no_cache_next:
            payload["cache_prompt"] = False
            self._force_no_cache_next = False

    @staticmethod
    def _ensure_usage(result: Dict[str, Any]) -> None:
        # qwen3vl_run.py's _debug_calc_speed indexes usage unconditionally.
        usage = result.setdefault("usage", {})
        usage.setdefault("prompt_tokens", 0)
        usage.setdefault("completion_tokens", 0)
        usage.setdefault(
            "total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]
        )
