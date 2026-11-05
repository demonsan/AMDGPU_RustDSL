"""Minimal ctypes binding of the HIP runtime (libamdhip64.so): module loading, launches, memory, events.

Only what the JIT needs; no third-party dependency. The library is located through ``AMDGPU_JIT_HIP_LIB`` (a path), else
``$ROCM_PATH/lib/libamdhip64.so``, else the default ROCm install paths, else the dynamic loader's search path.
"""
import ctypes
import os

_lib = None

HIP_MEMCPY_H2D = 1
HIP_MEMCPY_D2H = 2
HIP_MEMCPY_D2D = 3
# hipModuleLaunchKernel `extra` markers (hip_runtime_api.h, AMD platform)
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


class HipError(RuntimeError):
    pass


def _candidates():
    env = os.environ.get("AMDGPU_JIT_HIP_LIB")
    if env:
        yield env
    rocm = os.environ.get("ROCM_PATH")
    if rocm:
        yield os.path.join(rocm, "lib", "libamdhip64.so")
    for root in ("/shared/apps/ubuntu/opt/rocm-7.0.2", "/opt/rocm"):
        yield os.path.join(root, "lib", "libamdhip64.so")
    yield "libamdhip64.so"


def lib():
    """The loaded HIP runtime (loaded on first use)."""
    global _lib
    if _lib is not None:
        return _lib
    err = None
    for path in _candidates():
        if os.sep in path and not os.path.exists(path):
            continue
        try:
            l = ctypes.CDLL(path)
            break
        except OSError as e:  # pragma: no cover - environment dependent
            err = e
    else:
        raise HipError(f"libamdhip64.so not found (set AMDGPU_JIT_HIP_LIB): {err}")
    vp, u, sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t
    sig = {
        "hipGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
        "hipInit": ([u], ctypes.c_int),
        "hipSetDevice": ([ctypes.c_int], ctypes.c_int),
        "hipGetDevice": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        "hipModuleLoadData": ([ctypes.POINTER(vp), vp], ctypes.c_int),
        "hipModuleUnload": ([vp], ctypes.c_int),
        "hipModuleGetFunction": ([ctypes.POINTER(vp), vp, ctypes.c_char_p], ctypes.c_int),
        "hipModuleLaunchKernel": ([vp, u, u, u, u, u, u, u, vp, ctypes.POINTER(vp), ctypes.POINTER(vp)], ctypes.c_int),
        "hipMalloc": ([ctypes.POINTER(vp), sz], ctypes.c_int),
        "hipFree": ([vp], ctypes.c_int),
        "hipMemcpy": ([vp, vp, sz, ctypes.c_int], ctypes.c_int),
        "hipMemset": ([vp, ctypes.c_int, sz], ctypes.c_int),
        "hipDeviceSynchronize": ([], ctypes.c_int),
        "hipStreamSynchronize": ([vp], ctypes.c_int),
        "hipEventCreate": ([ctypes.POINTER(vp)], ctypes.c_int),
        "hipEventDestroy": ([vp], ctypes.c_int),
        "hipEventRecord": ([vp, vp], ctypes.c_int),
        "hipEventSynchronize": ([vp], ctypes.c_int),
        "hipEventElapsedTime": ([ctypes.POINTER(ctypes.c_float), vp, vp], ctypes.c_int),
    }
    for name, (args, res) in sig.items():
        f = getattr(l, name)
        f.argtypes = args
        f.restype = res
    _lib = l
    return l


def check(rc, what):
    if rc != 0:
        msg = lib().hipGetErrorString(rc)
        raise HipError(f"{what} failed: {rc} ({msg.decode() if msg else '?'})")


def malloc(nbytes):
    p = ctypes.c_void_p()
    check(lib().hipMalloc(ctypes.byref(p), max(1, int(nbytes))), "hipMalloc")
    return p.value


def free(ptr):
    check(lib().hipFree(ctypes.c_void_p(ptr)), "hipFree")


def memcpy_htod(dst, data):
    buf = (ctypes.c_char * len(data)).from_buffer_copy(bytes(data))
    check(lib().hipMemcpy(ctypes.c_void_p(dst), buf, len(data), HIP_MEMCPY_H2D), "hipMemcpy H2D")


def memcpy_dtoh(src, nbytes):
    buf = (ctypes.c_char * nbytes)()
    check(lib().hipMemcpy(buf, ctypes.c_void_p(src), nbytes, HIP_MEMCPY_D2H), "hipMemcpy D2H")
    return bytes(buf)


def memset(ptr, value, nbytes):
    check(lib().hipMemset(ctypes.c_void_p(ptr), value, nbytes), "hipMemset")


def synchronize(stream=None):
    if stream:
        check(lib().hipStreamSynchronize(ctypes.c_void_p(stream)), "hipStreamSynchronize")
    else:
        check(lib().hipDeviceSynchronize(), "hipDeviceSynchronize")


class Module:
    """A loaded code object (kept alive as long as functions of it are used)."""

    def __init__(self, image: bytes):
        self._image = ctypes.create_string_buffer(image, len(image))
        m = ctypes.c_void_p()
        check(lib().hipModuleLoadData(ctypes.byref(m), self._image), "hipModuleLoadData")
        self.handle = m.value
        self._funcs = {}

    def function(self, symbol: str):
        f = self._funcs.get(symbol)
        if f is None:
            h = ctypes.c_void_p()
            check(lib().hipModuleGetFunction(ctypes.byref(h), ctypes.c_void_p(self.handle), symbol.encode()), f"hipModuleGetFunction({symbol})")
            f = self._funcs[symbol] = h.value
        return f


def launch(func, grid, block, kernarg: bytes, stream=None):
    """``hipModuleLaunchKernel`` with the kernarg segment passed as one buffer (HIP_LAUNCH_PARAM_BUFFER_POINTER)."""
    buf = ctypes.create_string_buffer(kernarg, len(kernarg))
    size = ctypes.c_size_t(len(kernarg))
    extra = (ctypes.c_void_p * 5)(HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(buf, ctypes.c_void_p), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                  ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), HIP_LAUNCH_PARAM_END)
    gx, gy, gz = (tuple(grid) + (1, 1, 1))[:3]
    bx, by, bz = (tuple(block) + (1, 1, 1))[:3]
    check(lib().hipModuleLaunchKernel(ctypes.c_void_p(func), gx, gy, gz, bx, by, bz, 0, ctypes.c_void_p(stream or 0), None, extra), "hipModuleLaunchKernel")


class Event:
    def __init__(self):
        e = ctypes.c_void_p()
        check(lib().hipEventCreate(ctypes.byref(e)), "hipEventCreate")
        self.handle = e.value

    def record(self, stream=None):
        check(lib().hipEventRecord(ctypes.c_void_p(self.handle), ctypes.c_void_p(stream or 0)), "hipEventRecord")

    def synchronize(self):
        check(lib().hipEventSynchronize(ctypes.c_void_p(self.handle)), "hipEventSynchronize")

    def elapsed_ms(self, end):
        ms = ctypes.c_float()
        check(lib().hipEventElapsedTime(ctypes.byref(ms), ctypes.c_void_p(self.handle), ctypes.c_void_p(end.handle)), "hipEventElapsedTime")
        return ms.value

    def __del__(self):
        try:
            if self.handle and _lib is not None:
                _lib.hipEventDestroy(ctypes.c_void_p(self.handle))
        except Exception:
            pass
