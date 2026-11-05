"""Device tensor adapters: torch tensors, any object with ``__dlpack__``, our own ``DeviceBuffer`` and raw integer pointers.

Every argument is normalised to a ``TensorRef`` (pointer, byte size, shape, element strides or None, element size, dtype name). The ops check
shapes, element sizes, contiguity and alignment on that; nothing here copies data.
"""
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

from . import _hip

# element sizes accepted per logical dtype (fp8 data travels as any 1-byte type: torch float8_e4m3fn, uint8, int8)
_BYTE_DTYPES = {"float8_e4m3fn", "float8_e4m3fnuz", "uint8", "int8", "fp8", "e8m0", "u8"}
_BF16_DTYPES = {"bfloat16", "bf16", "uint16", "int16"}


@dataclass
class TensorRef:
    ptr: int
    nbytes: int
    shape: Tuple[int, ...]
    strides: Optional[Tuple[int, ...]]  # in elements; None = unknown (raw pointer)
    itemsize: int
    dtype: str
    owner: object = None  # keeps a dlpack capsule / torch tensor alive during the call
    capacity: Optional[int] = None  # bytes addressable from ptr (>= nbytes: e.g. a zeroed slack tail behind a view); None = nbytes

    @property
    def cap(self):
        return self.nbytes if self.capacity is None else self.capacity

    def contiguous(self):
        if self.strides is None:
            return True
        expect = 1
        for dim, st in zip(reversed(self.shape), reversed(self.strides)):
            if dim != 1 and st != expect:
                return False
            expect *= dim
        return True


class DeviceBuffer:
    """A ``hipMalloc`` allocation with a shape / dtype (for stdlib-only use: tests, scripts). ``free()`` or garbage collection releases it."""

    def __init__(self, shape, dtype="uint8", itemsize=None, nbytes=None, data=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.itemsize = itemsize or (2 if dtype in _BF16_DTYPES else 4 if dtype in ("float32", "int32", "uint32") else 1)
        n = 1
        for s in self.shape:
            n *= s
        self.nbytes = int(nbytes if nbytes is not None else n * self.itemsize)
        self.ptr = _hip.malloc(self.nbytes)
        if data is not None:
            self.upload(data)

    def upload(self, data: bytes, offset=0):
        if offset + len(data) > self.nbytes:
            raise ValueError("DeviceBuffer.upload: data larger than the buffer")
        _hip.memcpy_htod(self.ptr + offset, data)

    def download(self, nbytes=None) -> bytes:
        return _hip.memcpy_dtoh(self.ptr, self.nbytes if nbytes is None else nbytes)

    def fill(self, byte):
        _hip.memset(self.ptr, byte, self.nbytes)

    def free(self):
        if self.ptr:
            _hip.free(self.ptr)
            self.ptr = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class RawPtr:
    """A raw device pointer with explicit metadata: ``RawPtr(ptr, shape, itemsize=1, nbytes=None)``."""

    def __init__(self, ptr, shape, itemsize=1, nbytes=None, dtype=None):
        self.ptr, self.shape, self.itemsize = int(ptr), tuple(shape), itemsize
        n = itemsize
        for s in self.shape:
            n *= s
        self.nbytes = int(nbytes if nbytes is not None else n)
        self.dtype = dtype or ("uint8" if itemsize == 1 else "bfloat16" if itemsize == 2 else "float32")


# ---- DLPack (DLPack v0.x / 1.0 DLTensor layout; the capsule is read, never consumed) ----
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


_DL_CODES = {0: "int", 1: "uint", 2: "float", 4: "bfloat"}


def _from_dlpack(obj):
    cap = obj.__dlpack__()
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    p = get(cap, b"dltensor")
    t = ctypes.cast(p, ctypes.POINTER(_DLTensor)).contents  # DLManagedTensor starts with its DLTensor
    if t.device.device_type not in (2, 10):  # kDLCUDA / kDLROCM
        raise ValueError(f"tensor is not on a GPU (DLPack device type {t.device.device_type})")
    shape = tuple(t.shape[i] for i in range(t.ndim))
    strides = tuple(t.strides[i] for i in range(t.ndim)) if t.strides else None
    item = max(1, t.dtype.bits // 8) * t.dtype.lanes
    n = item
    for s in shape:
        n *= s
    dt = f"{_DL_CODES.get(t.dtype.code, 'dl' + str(t.dtype.code))}{t.dtype.bits}"
    return TensorRef(t.data + t.byte_offset, n, shape, strides, item, dt, owner=(obj, cap))


def as_ref(x) -> TensorRef:
    """Normalises a tensor-like argument."""
    if isinstance(x, TensorRef):
        return x
    if isinstance(x, (DeviceBuffer, RawPtr)):
        n = x.itemsize
        for s in x.shape:
            n *= s
        return TensorRef(x.ptr, n, x.shape, None, x.itemsize, x.dtype, owner=x, capacity=x.nbytes)
    mod = type(x).__module__
    if mod.startswith("torch") and hasattr(x, "data_ptr"):
        if not x.is_cuda:
            raise ValueError("torch tensor is not on the GPU")
        cap = x.untyped_storage().nbytes() - x.storage_offset() * x.element_size()
        return TensorRef(x.data_ptr(), x.numel() * x.element_size(), tuple(x.shape), tuple(x.stride()), x.element_size(),
                         str(x.dtype).replace("torch.", ""), owner=x, capacity=cap)
    if hasattr(x, "__dlpack__"):
        return _from_dlpack(x)
    raise TypeError(f"unsupported tensor argument {type(x).__name__}: pass a torch tensor, a __dlpack__ object, DeviceBuffer or RawPtr")


def expect(ref: TensorRef, name, shape, itemsize, min_bytes=None, align=16):
    """Shape / element size / contiguity / alignment checks of one argument (min_bytes: the allocation the kernel contract requires)."""
    if tuple(ref.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(ref.shape)}, expected {tuple(shape)}")
    if ref.itemsize != itemsize:
        raise ValueError(f"{name}: element size {ref.itemsize} bytes ({ref.dtype}), expected {itemsize}")
    if itemsize == 1 and ref.dtype not in _BYTE_DTYPES and not ref.dtype.endswith("8"):
        raise ValueError(f"{name}: dtype {ref.dtype} is not a 1-byte fp8/byte type")
    if not ref.contiguous():
        raise ValueError(f"{name}: must be contiguous (strides {ref.strides})")
    if ref.ptr % align:
        raise ValueError(f"{name}: pointer 0x{ref.ptr:x} not {align}-byte aligned")
    if min_bytes is not None and ref.cap < min_bytes:
        raise ValueError(f"{name}: {ref.cap} addressable bytes, the kernel contract needs at least {min_bytes}")


def current_stream(refs, stream):
    """The stream to launch on: the explicit handle, else torch's current stream when any argument is a torch tensor, else 0 (null stream)."""
    if stream is not None:
        return int(getattr(stream, "cuda_stream", stream))
    for r in refs:
        o = r.owner
        if type(o).__module__.startswith("torch"):
            import torch  # only reached when torch tensors were passed

            return torch.cuda.current_stream().cuda_stream
    return 0
