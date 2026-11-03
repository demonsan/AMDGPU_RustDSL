#!/usr/bin/env python3
"""Host test for tests/hostgen/json_kernarg.py (no GPU)."""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import json_kernarg  # noqa: E402


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def sample_values(meta):
    vals = {}
    for i, a in enumerate(meta["args"]):
        if a["type"] == "Ptr":
            vals[a["name"]] = 0x1000 + 0x100 * i
        elif a["type"] == "F32":
            vals[a["name"]] = 1.5
        elif a["type"] == "F16":
            vals[a["name"]] = 3
        else:
            vals[a["name"]] = i + 1
    return vals


def check(meta, expected_size):
    assert meta["schema_version"] == 1, "schema_version"
    assert meta["kernarg_size"] == expected_size, (meta["kernel"], meta["kernarg_size"])
    vals = sample_values(meta)
    data = json_kernarg.pack(meta, vals)
    assert len(data) == expected_size
    by_name = {a["name"]: a for a in meta["args"]}
    for name, value in vals.items():
        a = by_name[name]
        fmt = json_kernarg.TYPE_FMT[a["type"]]
        got = struct.unpack_from(fmt, data, a["offset"])[0]
        if a["type"] == "F32":
            assert abs(got - value) < 1e-6, (name, got, value)
        else:
            assert got == value, (name, got, value)
    print(f"OK {meta['kernel']}: {len(meta['args'])} args, {len(data)} bytes")
    return data


def main():
    files = [
        ("tests/gpu/generated/fmha_fwd_hd128_mxfp8_v3_vs128_abi.json", 656),
        ("tests/gpu/generated/fmha_fwd_hd128_mxfp8_v3_kvsplit_abi.json", 648),
        ("tests/gpu/generated_gemm/gemm_mxfp8_bs_v3_abi.json", 96),
        ("tests/gpu/generated_gemm/gemm_mxfp8_bs_v4_abi.json", 96),
    ]
    found = 0
    for rel, size in files:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"SKIP missing {rel}")
            continue
        check(load(rel), size)
        found += 1
    assert found > 0, "no generated JSON files found"
    print(f"JSON_KERNARG_OK ({found} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
