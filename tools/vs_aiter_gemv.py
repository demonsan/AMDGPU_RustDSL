#!/usr/bin/env python3
"""Small-M GEMM / GEMV against aiter's prebuilt gfx950 ASM GEMMs, same process, same data (docs/perf-vs-aiter-v12-mb.md).

The W6 grid of tools/perf_ab_cases.py (M in {1, 4, 16, 32, 64} x N = K in {4096, 7168, 8192}) plus the v9.2 rows (64 x 8192 x 8192, 128 x 4096 x 8192), for every dense operand kind:
ours = the kernel `amdc dispatch gemm` picks (own backend, with its split-K / reduce), run by tests/gpu/vendor_gemm_host; aiter = every code object of the family directory the host supports,
the fastest reported with its tile. Ratio = aiter time / ours (> 1: ours is faster). fp8 per-token x per-channel is compared with the int8 per-token kernels (same operand layout, as in v9.2).
bf16: aiter's split-K bf16 code objects (fp32 partials + their own reduce) are not driven by the host, so the bf16 column compares with the non-split 256x256 kernel and is NOT like-for-like.

    python3 tools/vs_aiter_gemv.py [--amdc PATH] [--reps 3] [--out DIR] [--kinds i8_rowcol,...]       (HIP_VISIBLE_DEVICES selects the GPU)
"""
import argparse, json, os, re, statistics, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HSA = os.environ.get('AITER_HSA', '/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950')
ROCM = os.environ.get('ROCM_PATH', '/shared/apps/ubuntu/opt/rocm-7.0.2')
CLANG = os.environ.get('CLANG', '/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang')

# (label, dispatch kind, ours host family, aiter dir, aiter host family)
KINDS = [
    ('int8 per-token', 'i8_rowcol', 'ours_v4_i8', 'i8gemm', 'asm_i8'),
    ('fp8 per-token x per-channel (vs the int8 kernels)', 'fp8_rowcol', 'ours_v4_rowcol_fp8', 'i8gemm', 'asm_i8'),
    ('fp8 blockscale', 'fp8_blockscale', 'ours_v4_fp8bs', 'fp8gemm_blockscale', 'asm_fp8bs'),
    ('bf16 (aiter: non-split 256x256 only)', 'bf16', 'ours_v4_bf16', 'bf16gemm', 'asm_bf16_pre'),
]
SHAPES = [(m, nk, nk) for nk in (4096, 7168, 8192) for m in (1, 4, 16, 32, 64)] + [(128, 4096, 8192)]
# v9.2 rows that exist for these kinds (op-coverage-design 7.4 split-K rows, 2026-11-08 / 09)
V92 = {('int8 per-token', (128, 4096, 8192)): 1.27, ('int8 per-token', (64, 8192, 8192)): 1.62,
       ('fp8 blockscale', (128, 4096, 8192)): 1.45, ('fp8 blockscale', (64, 8192, 8192)): 1.57,
       ('fp8 per-token x per-channel (vs the int8 kernels)', (128, 4096, 8192)): 1.87, ('fp8 per-token x per-channel (vs the int8 kernels)', (64, 8192, 8192)): 1.62,
       ('bf16 (aiter: non-split 256x256 only)', (128, 4096, 8192)): 3.9, ('bf16 (aiter: non-split 256x256 only)', (64, 8192, 8192)): 3.6}


def sh(cmd, env=None, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **{k: str(v) for k, v in (env or {}).items()}), timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def vname(path):
    m = re.search(rb'_ZN[0-9A-Za-z_]+E', open(path, 'rb').read())
    return m.group(0).decode() if m else None


def tile_of_name(n):
    m = re.search(r'_(\d+)x(\d+)', n)
    return (int(m.group(1)), int(m.group(2))) if m else None


def our_tile(kernel):
    if kernel.endswith(('_64', '_64_splitk', '_64_splitk_lc')):
        return 64, 128
    if kernel.endswith(('_128', '_128_splitk')) or kernel.startswith('gemm_a8w8_blockscale'):
        return 128, 128
    return 256, 256


def run_host(host, args, env, reps):
    """Median us over `reps` runs: (us, ok)."""
    ts, ok = [], True
    for _ in range(reps):
        rc, out = sh([host] + [str(a) for a in args], env)
        m = re.search(r': ([0-9.]+) us .* bad=(\d+)', out)
        if rc != 0 or not m:
            return None, False
        ts.append(float(m.group(1)))
        ok &= int(m.group(2)) == 0
    return statistics.median(ts), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amdc', default='/tmp/target-v12/release/amdc')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default='build/vs_aiter_gemv')
    ap.add_argument('--kinds', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    host = os.path.join(a.out, 'vendor_gemm_host')
    rc, out = sh([os.path.join(ROCM, 'bin', 'amdclang++'), '-x', 'hip', '-O2', '--offload-arch=gfx950', '--rocm-path=' + ROCM, '-std=c++17', '-I', os.path.join(ROOT, 'tests', 'gpu'),
                  os.path.join(ROOT, 'tests', 'gpu', 'vendor_gemm_host.cpp'), '-o', host, '-L' + os.path.join(ROCM, 'lib'), '-lamdhip64', '-Wl,-rpath,' + os.path.join(ROCM, 'lib')])
    if rc != 0:
        sys.exit('host build failed:\n' + out[-2000:])
    cos = {}

    def co_of(k):
        if k not in cos:
            s, c = os.path.join(a.out, k + '.s'), os.path.join(a.out, k + '.co')
            rc, out = sh([a.amdc, 'emit', k, '--backend', 'own', '--out', s])
            if rc != 0:
                sys.exit(f'emit {k}: {out[-500:]}')
            rc, out = sh([CLANG, '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mcode-object-version=6', '-o', c, s])
            if rc != 0:
                sys.exit(f'assemble {k}: {out[-500:]}')
            cos[k] = c
        return cos[k]

    want = set(x for x in a.kinds.split(',') if x)
    rows = []
    for label, kind, ofam, vdir, vfam in KINDS:
        if want and kind not in want:
            continue
        vcos = sorted(f for f in os.listdir(os.path.join(HSA, vdir)) if f.endswith('.co'))
        for (m, n, k) in SHAPES:
            rc, out = sh([a.amdc, 'dispatch', 'gemm', kind, str(m), str(n), str(k)])
            p = json.loads(out[out.index('{'):])
            kn, s, red = p['kernel'], p['splits'], p.get('reduce')
            env = dict(SPLITK=s, ITERS=200, WARMUP=50, CHECK_ROWS=min(m, 8))
            if red:
                env.update(REDUCE_CO=co_of(red), REDUCE_KN=red)
            tm, tn = our_tile(kn)
            ou, ook = run_host(host, [ofam, co_of(kn), kn, m, n, k, tm, tn], env, a.reps)
            best = None
            for f in vcos:
                t = tile_of_name(f)
                if not t or (vfam == 'asm_bf16_pre' and ('bpreshuffle' not in f or 'splitk' in f)):
                    continue
                path = os.path.join(HSA, vdir, f)
                vu, vok = run_host(host, [vfam, path, vname(path), m, n, k, t[0], t[1]], dict(ITERS=200, WARMUP=50, CHECK_ROWS=min(m, 8)), a.reps)
                if vu is not None and vok and (best is None or vu < best[0]):
                    best = (vu, f.replace('.co', ''))
            row = dict(family=label, shape=f'{m}x{n}x{k}', ours_us=ou, ours_ok=ook, ours_kernel=kn + (f' S={s}' if s > 1 else ''), aiter_us=best[0] if best else None,
                       aiter_kernel=best[1] if best else None, ratio=(best[0] / ou) if (ou and best) else None, v92=V92.get((label, (m, n, k))))
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(a.out, 'gemv_vs_aiter.json'), 'w'), indent=1)
    f = lambda x, d=1: '-' if x is None else f'{x:.{d}f}'
    print('\n| family | shape (M x N x K) | ours us (kernel) | aiter us (best tile) | ratio | v9.2 |\n|---|---|--:|--:|--:|--:|')
    for r in rows:
        print(f"| {r['family']} | {r['shape']} | {f(r['ours_us'], 2)} ({r['ours_kernel']}{'' if r['ours_ok'] else ', BAD'}) | {f(r['aiter_us'], 2)} ({r['aiter_kernel']}) | {f(r['ratio'], 2)} | {f(r['v92'], 2)} |")


if __name__ == '__main__':
    main()
