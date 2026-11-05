"""Host-only unit tests of amdgpu_jit (no GPU, no toolchain): expression evaluation, kernarg packing, contract validation, path selection.

    python3 -m unittest discover -s python/tests -p 'test_unit.py'
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amdgpu_jit._kernarg import ContractError, KernelInfo, eval_ext, fmt_ext, pack, pack_fast, scalars_to_ka, validate, validate_memo  # noqa: E402
from amdgpu_jit.ops import select_path  # noqa: E402


def info(spec=None):
    return KernelInfo({
        "schema": 1, "kernel": "toy", "symbol": "toy", "arch": "gfx950", "block": [256, 1, 1], "kernarg_size": 32, "kernarg_align": 8,
        "specialize": spec or {},
        "args": [{"name": "ptr_a", "offset": 0, "size": 8, "kind": "ptr", "specializable": False, "specialized": None},
                 {"name": "m", "offset": 8, "size": 4, "kind": "i32", "specializable": True, "specialized": (spec or {}).get("m")},
                 {"name": "k", "offset": 12, "size": 4, "kind": "u32", "specializable": True, "specialized": None},
                 {"name": "alpha", "offset": 16, "size": 4, "kind": "f32", "specializable": True, "specialized": None},
                 {"name": "_pad20", "offset": 20, "size": 12, "kind": "pad", "specializable": False, "specialized": None}],
        "contract": {"grid": [{"axis": 0, "ext": {"ceil_div": [{"ka": 8}, 256]}}],
                     "ka_is": [{"offset": 12, "ext": {"mul": [{"sym": "KT"}, {"c": 128}]}}],
                     "ka_range": [{"offset": 8, "lo": 1, "hi": None}], "ka_set": [],
                     "tensors": [{"arg": "ptr_a", "alloc_bytes": {"mul": [{"ka": 8}, {"ka": 12}]}, "logical_bytes": None}],
                     "syms": [{"name": "KT", "lo": 1, "hi": 64}]}})


class T(unittest.TestCase):
    def test_eval(self):
        e = {"add": [{"ceil_div": [{"ka": 8}, 256]}, {"sub": [{"grid": 1}, {"sym": "X"}]}]}
        self.assertEqual(eval_ext(e, {8: 257}, (1, 7, 1), {"X": 2}), 2 + 5)
        self.assertEqual(eval_ext({"floor_div": [{"c": -7}, 2]}, {}, (1, 1, 1), {}), -4)
        self.assertIn("ceil", fmt_ext(e))

    def test_pack_and_validate(self):
        i = info()
        buf, ka = pack(i, {"ptr_a": 0x1000, "m": 512, "k": 256, "alpha": 0.5})
        self.assertEqual(len(buf), 32)
        self.assertEqual(struct.unpack_from("<QiIf", buf, 0), (0x1000, 512, 256, 0.5))
        validate(i, ka, (2, 1, 1), {"KT": 2}, {"ptr_a": 512 * 256})
        with self.assertRaises(ContractError):
            validate(i, ka, (3, 1, 1), {"KT": 2}, {"ptr_a": 512 * 256})  # grid
        with self.assertRaises(ContractError):
            validate(i, ka, (2, 1, 1), {"KT": 3}, {"ptr_a": 512 * 256})  # ka_is
        with self.assertRaises(ContractError):
            validate(i, ka, (2, 1, 1), {"KT": 2}, {"ptr_a": 100})  # tensor size
        with self.assertRaises(ContractError):
            validate(i, ka, (2, 1, 1), {"KT": 99}, {"ptr_a": 1 << 40})  # symbol range (and ka_is)

    def test_pack_fast_matches_pack(self):
        i = info()
        v = {"ptr_a": 0x1000, "m": 512, "k": 256, "alpha": 0.5}
        b1, ka = pack(i, v)
        b2, sc = pack_fast(i, v)
        self.assertEqual(b1, b2)
        self.assertEqual(scalars_to_ka(i, sc), ka)
        validate_memo(i, sc, (2, 1, 1), {"KT": 2}, {"ptr_a": 512 * 256})
        validate_memo(i, sc, (2, 1, 1), {"KT": 2}, {"ptr_a": 512 * 256})  # memo hit
        with self.assertRaises(ContractError):
            validate_memo(i, sc, (3, 1, 1), {"KT": 2}, {"ptr_a": 512 * 256})
        with self.assertRaises(ContractError):
            pack_fast(i, {"ptr_a": 1, "m": 1, "k": -1, "alpha": 1.0})
        with self.assertRaises(ContractError):
            pack_fast(info({"m": 512}), {"ptr_a": 1, "m": 256, "k": 1, "alpha": 1.0})

    def test_pack_errors(self):
        with self.assertRaises(ContractError):
            pack(info(), {"ptr_a": 1, "m": 1, "k": 1})  # missing alpha
        with self.assertRaises(ContractError):
            pack(info(), {"ptr_a": 1, "m": 1, "k": 1, "alpha": 1.0, "zz": 3})  # unknown
        with self.assertRaises(ContractError):
            pack(info({"m": 512}), {"ptr_a": 1, "m": 256, "k": 1, "alpha": 1.0})  # specialized mismatch
        with self.assertRaises(ContractError):
            pack(info(), {"ptr_a": 1, "m": 1, "k": -1, "alpha": 1.0})  # u32 range

    def test_select_path(self):
        self.assertEqual(select_path(4, 8, 2048, 2048), ("vs128", 0))
        self.assertEqual(select_path(1, 8, 8192, 8192)[0], "vs128")
        self.assertEqual(select_path(2, 8, 2048, 2048), ("t128", 0))
        self.assertEqual(select_path(1, 8, 2048, 2048), ("kvsplit", 4))


if __name__ == "__main__":
    unittest.main()
