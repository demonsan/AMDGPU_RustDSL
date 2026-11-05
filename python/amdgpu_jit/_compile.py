"""JIT compilation: ``amdc emit`` (+ ``amdc jit-info``) -> ROCm clang -> code object, with a disk cache and an in-process module cache.

Toolchain discovery
  amdc   : ``AMDGPU_JIT_AMDC``, else ``<repo>/target/release/amdc``, ``<repo>/target/debug/amdc``, ``$AMD_CARGO_TARGET/{release,debug}/amdc``.
  clang  : ``AMDGPU_JIT_CLANG``, else ``$ASM_ROC/llvm/bin/clang``, ``$ROCM_PATH/llvm/bin/clang``, the default ROCm installs.
  arch   : ``AMDGPU_JIT_ARCH`` (default gfx950).
  backend: ``AMDGPU_JIT_BACKEND`` (``own`` (default) or ``llvm-tail``).
Disk cache: ``AMDGPU_JIT_CACHE_DIR`` (default ``~/.cache/amdgpu_jit``), one directory per key holding ``kernel.co``, ``info.json`` and
``kernel.s``; the key hashes kernel name, specialization, backend, arch, the amdc binary (content hash), the clang version and the package's
cache format. Entries are written to a temporary directory and renamed into place (atomic; concurrent compilers of the same key are safe).
``AMDGPU_JIT_CACHE=0`` disables the disk cache (the in-process cache stays).
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

from . import _hip
from ._kernarg import KernelInfo, pack, validate

CACHE_FORMAT = 1
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lock = threading.Lock()
_modules = {}  # key -> Kernel
_file_hash = {}  # (path, size, mtime) -> sha256
stats = {"memory_hits": 0, "disk_hits": 0, "compiles": 0, "compile_seconds": 0.0}


class CompileError(RuntimeError):
    pass


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def amdc_path():
    p = os.environ.get("AMDGPU_JIT_AMDC")
    if p:
        return p
    tgt = os.environ.get("AMD_CARGO_TARGET") or os.environ.get("CARGO_TARGET_DIR")
    cands = [os.path.join(_REPO, "target", "release", "amdc"), os.path.join(_REPO, "target", "debug", "amdc")]
    if tgt:
        cands = [os.path.join(tgt, "release", "amdc"), os.path.join(tgt, "debug", "amdc")] + cands
    p = _first_existing(cands)
    if not p:
        raise CompileError("amdc not found: build it (cargo build --release --bin amdc) or set AMDGPU_JIT_AMDC")
    return p


def clang_path():
    p = os.environ.get("AMDGPU_JIT_CLANG")
    if p:
        return p
    roots = [os.environ.get("ASM_ROC"), os.environ.get("ROCM_PATH"), "/shared/apps/ubuntu/opt/rocm-10.1.0", "/opt/rocm"]
    p = _first_existing([os.path.join(r, "llvm", "bin", "clang") for r in roots if r])
    if not p:
        raise CompileError("ROCm clang not found: set AMDGPU_JIT_CLANG")
    return p


def arch():
    return os.environ.get("AMDGPU_JIT_ARCH", "gfx950")


def backend():
    return os.environ.get("AMDGPU_JIT_BACKEND", "own")


def _hash_file(path):
    st = os.stat(path)
    k = (path, st.st_size, st.st_mtime_ns)
    h = _file_hash.get(k)
    if h is None:
        d = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                d.update(chunk)
        h = _file_hash[k] = d.hexdigest()[:16]
    return h


_clang_ver = {}


def _clang_version(clang):
    v = _clang_ver.get(clang)
    if v is None:
        out = subprocess.run([clang, "--version"], capture_output=True, text=True, check=True).stdout
        v = _clang_ver[clang] = out.splitlines()[0].strip() if out else "unknown"
    return v


def cache_dir():
    return os.path.expanduser(os.environ.get("AMDGPU_JIT_CACHE_DIR", os.path.join("~", ".cache", "amdgpu_jit")))


def spec_string(spec):
    return ",".join(f"{k}={spec[k]}" for k in sorted(spec))


def cache_key(kernel, spec, amdc, clang):
    d = {"format": CACHE_FORMAT, "kernel": kernel, "specialize": spec_string(spec), "backend": backend(), "arch": arch(),
         "amdc": _hash_file(amdc), "clang": _clang_version(clang)}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:24], d


class Kernel:
    """A compiled, loaded kernel: its jit-info, the HIP function and a launch helper (pack + validate + launch)."""

    def __init__(self, key, info: KernelInfo, image: bytes, path=None):
        self.key, self.info, self.path = key, info, path
        self.module = _hip.Module(image)
        self.func = self.module.function(info.symbol)

    def kernarg(self, values, grid, syms, sizes, check=True):
        buf, ka = pack(self.info, values)
        if check:
            validate(self.info, ka, grid, syms, sizes)
        return buf

    def launch(self, values, grid, syms=None, sizes=None, stream=None, check=True):
        buf = self.kernarg(values, grid, syms or {}, sizes or {}, check)
        _hip.launch(self.func, grid, self.info.block, buf, stream)

    def launch_raw(self, kernarg: bytes, grid, stream=None):
        _hip.launch(self.func, grid, self.info.block, kernarg, stream)


def _run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise CompileError(f"{what} failed (exit {r.returncode}): {' '.join(cmd)}\n{r.stderr[-4000:]}{r.stdout[-2000:]}")
    return r.stdout


def _compile_into(d, kernel, spec, amdc, clang):
    """amdc emit + jit-info (concurrently) and clang into directory d."""
    sp = ["--specialize", spec_string(spec)] if spec else []
    s_path, co_path = os.path.join(d, "kernel.s"), os.path.join(d, "kernel.co")
    emit = subprocess.Popen([amdc, "emit", kernel, "--backend", backend(), "--out", s_path] + sp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = subprocess.Popen([amdc, "jit-info", kernel, "--backend", backend()] + sp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    eo, ee = emit.communicate()
    io_, ie = info.communicate()
    if emit.returncode != 0:
        raise CompileError(f"amdc emit {kernel} {' '.join(sp)} failed (exit {emit.returncode}):\n{ee[-4000:]}{eo[-1000:]}")
    if info.returncode != 0:
        raise CompileError(f"amdc jit-info {kernel} {' '.join(sp)} failed (exit {info.returncode}):\n{ie[-4000:]}")
    _run([clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch()}", "-mcode-object-version=6", "-o", co_path, s_path], "clang")
    d_info = json.loads(io_)
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(d_info, f, indent=1, sort_keys=True)
    return d_info


def get_kernel(kernel, specialize=None):
    """The loaded kernel `kernel` specialized for `specialize` (dict arg name -> value), compiling on a cache miss."""
    spec = {k: v for k, v in (specialize or {}).items()}
    amdc, clang = amdc_path(), clang_path()
    key, meta = cache_key(kernel, spec, amdc, clang)
    with _lock:
        k = _modules.get(key)
        if k is not None:
            stats["memory_hits"] += 1
            return k
        use_disk = os.environ.get("AMDGPU_JIT_CACHE", "1") != "0"
        root = cache_dir()
        final = os.path.join(root, key)
        if use_disk and os.path.isfile(os.path.join(final, "kernel.co")) and os.path.isfile(os.path.join(final, "info.json")):
            stats["disk_hits"] += 1
        else:
            t0 = time.perf_counter()
            os.makedirs(root, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix=f".{key}.", dir=root)
            try:
                _compile_into(tmp, kernel, spec, amdc, clang)
                with open(os.path.join(tmp, "key.json"), "w") as f:
                    json.dump(meta, f, indent=1, sort_keys=True)
                if use_disk:
                    try:
                        os.rename(tmp, final)  # atomic publish; another process may have won the race
                    except OSError:
                        if not os.path.isfile(os.path.join(final, "kernel.co")):
                            raise
                        shutil.rmtree(tmp, ignore_errors=True)
                else:
                    final = tmp
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            stats["compiles"] += 1
            stats["compile_seconds"] += time.perf_counter() - t0
        with open(os.path.join(final, "info.json")) as f:
            info = KernelInfo(json.load(f))
        with open(os.path.join(final, "kernel.co"), "rb") as f:
            image = f.read()
        if not use_disk:
            shutil.rmtree(final, ignore_errors=True)
        k = _modules[key] = Kernel(key, info, image, final if use_disk else None)
        return k


def clear_memory_cache():
    with _lock:
        _modules.clear()
