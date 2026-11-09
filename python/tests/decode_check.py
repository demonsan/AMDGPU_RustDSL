"""Node check of ``amdgpu_jit.paged_decode_bf16`` (torch-free, stdlib only): random bf16 Q / paged K / V, shuffled block tables and random
context lengths in [CTX / 2, CTX] (the data model of tests/gpu/pa_host.cpp), the op on the GPU, and pa_host's CPU reference (f64 softmax over
the bf16 values; pass = |got - ref| <= 1e-2 |ref| + 2e-2 per element). Covers both dispatch forms: last-arriver `_c` (small B HK) and fused.

    python3 python/tests/decode_check.py              # the default cases
    python3 python/tests/decode_check.py B H HK CTX   # one case
Doubles as the example of driving the op without torch (DeviceBuffer + bytes).
"""
import math
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amdgpu_jit as aj  # noqa: E402

D, P = 128, 16


def bf16_bits(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_val(h):
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def case(B, H, HK, CTX, seed=1, sample_heads=None):
    rnd = random.Random(seed)
    G = H // HK
    maxp = (CTX + P - 1) // P
    npages = B * maxp
    lens = [rnd.randint(max(1, CTX // 2), CTX) for _ in range(B)]
    pages = list(range(npages))
    rnd.shuffle(pages)
    bt = [pages[b * maxp:(b + 1) * maxp] for b in range(B)]
    nq, nk = B * H * D, npages * P * HK * D
    q = [bf16_bits(rnd.gauss(0, 1)) for _ in range(nq)]
    kc = [bf16_bits(rnd.gauss(0, 1)) for _ in range(nk)]
    vc = [bf16_bits(rnd.gauss(0, 1)) for _ in range(nk)]
    dq = aj.DeviceBuffer((B, H, D), "bfloat16", data=struct.pack(f"<{nq}H", *q))
    dk = aj.DeviceBuffer((npages, P, HK, D), "bfloat16", data=struct.pack(f"<{nk}H", *kc))
    dv = aj.DeviceBuffer((npages, P, HK, D), "bfloat16", data=struct.pack(f"<{nk}H", *vc))
    dbt = aj.DeviceBuffer((B, maxp), "int32", data=struct.pack(f"<{B * maxp}i", *[x for row in bt for x in row]))
    dcl = aj.DeviceBuffer((B,), "int32", data=struct.pack(f"<{B}i", *lens))
    dout = aj.DeviceBuffer((B, H, D), "bfloat16")
    dout.fill(0)
    name, S = aj.paged_decode_bf16(dq, dk, dv, dbt, dcl, dout, max_context=CTX)
    aj.hip.synchronize()
    # timing (warm, the same launch)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        aj.paged_decode_bf16(dq, dk, dv, dbt, dcl, dout, max_context=CTX, check=False)
    aj.hip.synchronize()
    us = (time.perf_counter() - t0) / reps * 1e6
    out = struct.unpack(f"<{nq}H", dout.download())
    scale = 1.0 / math.sqrt(D)
    kv = lambda b, key, hk: ((bt[b][key // P] * P + key % P) * HK + hk) * D
    bad, maxe, checked = 0, 0.0, 0
    heads = range(H) if sample_heads is None else sorted(rnd.sample(range(H), sample_heads))
    for b in range(B):
        for h in heads:
            hk = h // G
            qv = [bf16_val(x) for x in q[(b * H + h) * D:(b * H + h + 1) * D]]
            s = []
            for key in range(lens[b]):
                base = kv(b, key, hk)
                s.append(sum(qv[d] * bf16_val(kc[base + d]) for d in range(D)) * scale)
            m = max(s)
            p = [math.exp(x - m) for x in s]
            l = sum(p)
            o = [0.0] * D
            for key, w in enumerate(p):
                base = kv(b, key, hk)
                for d in range(D):
                    o[d] += w * bf16_val(vc[base + d])
            for d in range(D):
                ref = o[d] / l
                got = bf16_val(out[(b * H + h) * D + d])
                e = abs(got - ref)
                maxe = max(maxe, e)
                checked += 1
                if e > 1e-2 * abs(ref) + 2e-2:
                    bad += 1
    ok = bad == 0
    print(f"{'ok  ' if ok else 'FAIL'} paged_decode_bf16 B={B} H={H} HK={HK} CTX={CTX}: {name} S={S}, {us:.1f} us/launch (incl. Python), "
          f"{checked} outputs checked, bad={bad} maxerr={maxe:.3e}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 5:
        sys.exit(0 if case(*map(int, sys.argv[1:])) else 1)
    oks = [case(2, 16, 2, 300), case(4, 64, 8, 1024, sample_heads=6), case(16, 64, 8, 512, sample_heads=3), case(32, 64, 8, 512, sample_heads=2)]
    sys.exit(0 if all(oks) else 1)
