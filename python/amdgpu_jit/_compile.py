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
Kernels from DSL IR text: ``get_kernel(..., ir_path="k.ir")`` compiles an ``irv1`` file (``amdc print-ir --stage dsl``, edited; docs/ir-text.md)
with ``amdc emit --from-ir`` (strictly verified); the cache key includes the SHA-256 of the file's text.
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
from ._kernarg import KernelInfo, pack_fast, validate_memo

CACHE_FORMAT = 1
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lock = threading.Lock()
_modules = {}  # key -> Kernel
_key_locks = {}  # key -> Lock: one compile per key, different keys concurrently
_fast = {}  # (kernel, spec items, backend, arch, cache flag) -> Kernel: the per-call lookup (no stat / hashing once loaded)
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


_FP = None


def _fingerprints():
    """Persistent toolchain fingerprints (<cache>/toolchain.json: file hash / clang version keyed by path, size and mtime), so a fresh process
    does not re-hash the amdc binary or re-run clang --version."""
    global _FP
    if _FP is None:
        try:
            with open(os.path.join(cache_dir(), "toolchain.json")) as f:
                _FP = json.load(f)
        except (OSError, ValueError):
            _FP = {}
    return _FP


def _fp_store():
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        tmp = os.path.join(cache_dir(), f".toolchain.{os.getpid()}.json")
        with open(tmp, "w") as f:
            json.dump(_FP, f, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(cache_dir(), "toolchain.json"))
    except OSError:
        pass


def _fp_key(kind, path):
    st = os.stat(path)
    return f"{kind}:{path}:{st.st_size}:{st.st_mtime_ns}"


def _hash_file(path):
    k = _fp_key("sha256", path)
    fp = _fingerprints()
    h = fp.get(k)
    if h is None:
        d = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                d.update(chunk)
        h = fp[k] = d.hexdigest()[:16]
        _fp_store()
    return h


def _clang_version(clang):
    k = _fp_key("version", clang)
    fp = _fingerprints()
    v = fp.get(k)
    if v is None:
        out = subprocess.run([clang, "--version"], capture_output=True, text=True, check=True).stdout
        v = fp[k] = out.splitlines()[0].strip() if out else "unknown"
        _fp_store()
    return v


def cache_dir():
    return os.path.expanduser(os.environ.get("AMDGPU_JIT_CACHE_DIR", os.path.join("~", ".cache", "amdgpu_jit")))


def spec_string(spec):
    return ",".join(f"{k}={spec[k]}" for k in sorted(spec))


def _ir_sha(ir_path):
    with open(ir_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:24]


def cache_key(kernel, spec, amdc, clang, ir_path=None):
    d = {"format": CACHE_FORMAT, "kernel": kernel, "specialize": spec_string(spec), "backend": backend(), "arch": arch(),
         "amdc": _hash_file(amdc), "clang": _clang_version(clang) if clang else "none"}
    if ir_path:
        d["ir_sha"] = _ir_sha(ir_path)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:24], d


class Kernel:
    """A compiled, loaded kernel: its jit-info, the HIP function and a launch helper (pack + validate + launch)."""

    def __init__(self, key, info: KernelInfo, image: bytes, path=None):
        self.key, self.info, self.path = key, info, path
        self.module = _hip.Module(image)
        self.func = self.module.function(info.symbol)

    def kernarg(self, values, grid, syms, sizes, check=True):
        buf, scalars = pack_fast(self.info, values)
        if check:
            validate_memo(self.info, scalars, grid, syms, sizes)
        return buf

    def launch(self, values, grid, syms=None, sizes=None, stream=None, check=True):
        buf = self.kernarg(values, grid, syms or {}, sizes or {}, check)
        _hip.launch(self.func, grid, self.info.block, buf, stream)

    def prepare(self, values, grid, syms=None, sizes=None, check=True, result=None):
        """Pack + validate once; the returned ``_hip.Prepared`` relaunches with one HIP call."""
        return _hip.Prepared(self.func, grid, self.info.block, self.kernarg(values, grid, syms or {}, sizes or {}, check), result)

    def launch_raw(self, kernarg: bytes, grid, stream=None):
        _hip.launch(self.func, grid, self.info.block, kernarg, stream)


def _run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise CompileError(f"{what} failed (exit {r.returncode}): {' '.join(cmd)}\n{r.stderr[-4000:]}{r.stdout[-2000:]}")
    return r.stdout


# ---------------------------------------------------------------------------------------------------------------------------- the persistent compile server

def _esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def _unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"n": "\n", "t": "\t"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


class _Server:
    """`amdc serve`: one persistent compiler process (the process start and the registry are paid once); requests are lines on its stdin, responses lines on its stdout, matched by id."""

    def __init__(self, amdc):
        self.proc = subprocess.Popen([amdc, "serve", "--jobs", str(os.cpu_count() or 8)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.wlock = threading.Lock()
        self.pending = {}
        self.next_id = 0
        self.dead = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            slot = self.pending.pop(parts[0], None)
            if slot is not None:
                slot[1] = (int(parts[1]), _unesc(parts[2]))
                slot[0].set()
        self.dead = True
        for slot in list(self.pending.values()):
            slot[0].set()

    def request(self, fields, timeout=600):
        if self.dead or self.proc.poll() is not None:
            return None
        with self.wlock:
            self.next_id += 1
            rid = str(self.next_id)
            slot = [threading.Event(), None]
            self.pending[rid] = slot
            try:
                self.proc.stdin.write("\t".join([rid] + [_esc(f) if f else "-" for f in fields]) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.dead = True
                self.pending.pop(rid, None)
                return None
        slot[0].wait(timeout)
        return slot[1]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


_servers = {}


def _server_for(amdc):
    if os.environ.get("AMDGPU_JIT_SERVER", "1") == "0" or os.environ.get("AMDGPU_JIT_ASSEMBLER", "inproc") == "clang":
        return None
    with _lock:
        s = _servers.get(amdc)
        if s is None or s.dead:
            try:
                s = _servers[amdc] = _Server(amdc)
            except OSError:
                return None
        return s


def close_servers():
    for s in list(_servers.values()):
        s.close()
    _servers.clear()


import atexit  # noqa: E402

atexit.register(close_servers)


def template_dir(amdc):
    """Shape templates built offline (`amdc template-build --template-dir`): `<cache>/templates/<sha256(amdc)[:24]>`."""
    return os.path.join(cache_dir(), "templates", _hash_file(amdc))


def cert_cache_dir(amdc):
    """The content-addressed certificate cache of this compiler (`amdc certify-batch` pre-populates the same directory)."""
    return os.path.join(cache_dir(), "certs", _hash_file(amdc))



def _compile_into(d, kernel, spec, amdc, clang, ir_path=None):
    """amdc emit + jit-info (concurrently) and clang into directory d."""
    sp = ["--specialize", spec_string(spec)] if spec else []
    if ir_path:
        sp += ["--from-ir", ir_path]
    s_path, co_path = os.path.join(d, "kernel.s"), os.path.join(d, "kernel.co")
    certs = cert_cache_dir(amdc)
    srv = _server_for(amdc)
    if srv is not None:
        # the persistent server: no process start, one compile, the certificate cached by content
        r = srv.request([kernel, spec_string(spec) if spec else "", backend(), s_path, co_path, os.path.join(d, "info.json"), ir_path or "", certs, "", template_dir(amdc)])
        if r is not None:
            code, msg = r
            if code == 0 or code == 3:
                if code == 3:
                    if not clang:
                        raise CompileError("the in-process assembler cannot encode this kernel and ROCm clang was not found (set AMDGPU_JIT_CLANG)")
                    _run([clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch()}", "-mcode-object-version=6", "-o", co_path, s_path], "clang")
                with open(os.path.join(d, "info.json")) as f:
                    return json.load(f)
            raise CompileError(f"amdc serve {kernel} {' '.join(sp)} failed (code {code}):\n{msg[-4000:]}")
    # the in-process assembler (amdc emit --co): no clang; AMDGPU_JIT_ASSEMBLER=clang forces the external assembler
    inproc = os.environ.get("AMDGPU_JIT_ASSEMBLER", "inproc") != "clang"
    emit = subprocess.Popen([amdc, "emit", kernel, "--backend", backend(), "--out", s_path] + (["--co", co_path, "--info", os.path.join(d, "info.json"), "--cert-cache", certs, "--template-dir", template_dir(amdc)] if inproc else []) + sp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = None if inproc else subprocess.Popen([amdc, "jit-info", kernel, "--backend", backend()] + sp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    eo, ee = emit.communicate()
    io_, ie = info.communicate() if info else (None, "")
    # exit 3: the in-process assembler cannot encode this kernel (the asm is written): fall back to clang below
    use_clang = (not inproc) or emit.returncode == 3
    if emit.returncode not in (0, 3):
        raise CompileError(f"amdc emit {kernel} {' '.join(sp)} failed (exit {emit.returncode}):\n{ee[-4000:]}{eo[-1000:]}")
    if info is not None and info.returncode != 0:
        raise CompileError(f"amdc jit-info {kernel} {' '.join(sp)} failed (exit {info.returncode}):\n{ie[-4000:]}")
    if use_clang:
        if not clang:
            raise CompileError("the in-process assembler cannot encode this kernel and ROCm clang was not found (set AMDGPU_JIT_CLANG)")
        _run([clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch()}", "-mcode-object-version=6", "-o", co_path, s_path], "clang")
    if io_ is None:  # one-process path: amdc wrote the info next to the code object
        with open(os.path.join(d, "info.json")) as f:
            return json.load(f)
    d_info = json.loads(io_)
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(d_info, f, indent=1, sort_keys=True)
    return d_info


def get_kernel(kernel, specialize=None, ir_path=None):
    """The loaded kernel `kernel` specialized for `specialize` (dict arg name -> value), compiling on a cache miss. With `ir_path` the kernel
    is compiled from that irv1 DSL IR file (its `dsl "name"` is the symbol; `kernel` is only a label)."""
    spec = {k: v for k, v in (specialize or {}).items()}
    irk = (os.path.abspath(ir_path), os.stat(ir_path).st_mtime_ns) if ir_path else None
    fk = (kernel, tuple(sorted(spec.items())), os.environ.get("AMDGPU_JIT_BACKEND"), os.environ.get("AMDGPU_JIT_ARCH"), os.environ.get("AMDGPU_JIT_AMDC"), irk)
    k = _fast.get(fk)
    if k is not None:
        stats["memory_hits"] += 1
        return k
    k = _get_kernel(kernel, spec, ir_path)
    _fast[fk] = k
    return k


def _get_kernel(kernel, spec, ir_path=None, load=True):
    """The loaded kernel; with `load=False` only make sure the disk-cache entry exists (compile + certificate on a miss) and return its directory: `amdgpu_jit.prewarm`."""
    amdc = amdc_path()
    try:
        clang = clang_path()
    except CompileError:
        clang = None  # the in-process assembler needs no clang
    key, meta = cache_key(kernel, spec, amdc, clang, ir_path)
    with _lock:
        k = _modules.get(key)
        if k is not None:
            stats["memory_hits"] += 1
            return k
        klock = _key_locks.setdefault(key, threading.Lock())
    # one lock per key: different kernels / shapes compile concurrently (amdc runs as a subprocess), the same key compiles once
    with klock:
        with _lock:
            k = _modules.get(key)
            if k is not None:
                stats["memory_hits"] += 1
                return k if load else k.path
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
                _compile_into(tmp, kernel, spec, amdc, clang, ir_path)
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
            with _lock:
                stats["compiles"] += 1
                stats["compile_seconds"] += time.perf_counter() - t0
        if not load:
            return final
        with open(os.path.join(final, "info.json")) as f:
            info = KernelInfo(json.load(f))
        with open(os.path.join(final, "kernel.co"), "rb") as f:
            image = f.read()
        if not use_disk:
            shutil.rmtree(final, ignore_errors=True)
        k = Kernel(key, info, image, final if use_disk else None)
        with _lock:
            _modules[key] = k
        return k


def clear_memory_cache():
    with _lock:
        _modules.clear()
        _fast.clear()
