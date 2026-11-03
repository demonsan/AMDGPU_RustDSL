#!/usr/bin/env python3
"""Read an `amdc gen-host` JSON ABI and pack a kernarg byte buffer. Host-only, stdlib only.

This is the Python/aiter counterpart of the generated C++ `Args`: it uses the same
`args[].offset/size/type` facts, so it cannot disagree with the C++ header about the layout.
It does not launch anything (no GPU required).

Example:
    python3 tests/hostgen/json_kernarg.py tests/gpu/generated_gemm/gemm_mxfp8_bs_v3_abi.json \
        --out /tmp/gemm.karg --set ptr_a=0x1000 --set ptr_b=0x2000 --set m=1280
"""
import argparse
import json
import struct
import sys

TYPE_FMT = {
    "Ptr": "<Q",   # 64-bit host pointer value
    "I32": "<i",
    "U32": "<I",
    "I1": "<I",
    "F32": "<f",
    "F16": "<H",
}


def _check(meta):
    if meta.get("schema_version") != 1:
        raise ValueError("unsupported/absent JSON schema_version (expected 1)")
    size = meta["kernarg_size"]
    occupied = bytearray(size)
    for a in meta["args"]:
        off, sz = a["offset"], a["size"]
        if off < 0 or sz <= 0 or off + sz > size:
            raise ValueError(f"{a['name']}: [{off},{off+sz}) outside kernarg[{size}]")
        for i in range(off, off + sz):
            if occupied[i]:
                raise ValueError(f"{a['name']} overlaps another argument at {i}")
            occupied[i] = 1
    return size


def pack(meta, values):
    """Build the kernarg buffer; missing args stay zero. Raises on type/size mismatch."""
    size = _check(meta)
    buf = bytearray(size)
    by_name = {a["name"]: a for a in meta["args"]}
    for name, value in values.items():
        a = by_name.get(name)
        if a is None:
            raise KeyError(f"{name}: not in {meta.get('kernel')} ABI")
        ty, off, sz = a["type"], a["offset"], a["size"]
        fmt = TYPE_FMT.get(ty)
        if fmt is not None:
            try:
                struct.pack_into(fmt, buf, off, value)
            except struct.error as e:
                raise ValueError(f"{name}={value!r} does not fit {ty} ({sz} bytes): {e}") from e
        else:
            raw = bytes(value) if not isinstance(value, str) else bytes.fromhex(value)
            if len(raw) != sz:
                raise ValueError(f"{name}: expected {sz} bytes, got {len(raw)}")
            buf[off:off + sz] = raw
    return bytes(buf)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metadata", help="*_abi.json from amdc gen-host")
    ap.add_argument("--out", help="write packed kernarg bytes here (default: summary only)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="scalar/pointer value; repeatable")
    ns = ap.parse_args(argv)
    meta = json.load(open(ns.metadata, encoding="utf-8"))
    vals = {}
    for item in ns.set:
        name, _, val = item.partition("=")
        vals[name] = int(val, 0)
    data = pack(meta, vals)
    print(f"{meta['kernel']}: kernarg_size={meta['kernarg_size']} bytes, args={len(meta['args'])}")
    for a in meta["args"]:
        if a["name"] in vals:
            print(f"  {a['name']:<18} off=0x{a['offset']:02X} size={a['size']} value={vals[a['name']]}")
    if ns.out:
        with open(ns.out, "wb") as f:
            f.write(data)
        print(f"wrote {ns.out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
