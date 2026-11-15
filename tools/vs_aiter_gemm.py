#!/usr/bin/env python3
"""Same-process, same-data GEMM comparison against aiter's prebuilt gfx950 ASM GEMMs (docs/op-coverage-design.md section 7.4 summary shapes).

For every (family, shape): ours = the kernel `amdc dispatch gemm` picks (own backend, with its split-K / reduce), run by tests/gpu/vendor_gemm_host;
aiter = every code object of the family's directory the host supports, the fastest one reported (its tile in the table). Ratio = aiter time / ours.

    python3 tools/vs_aiter_gemm.py [--amdc PATH] [--reps 3] [--out DIR]       (HIP_VISIBLE_DEVICES selects the GPU)
"""
import argparse, json, os, re, statistics, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HSA = os.environ.get('AITER_HSA', '/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950')
ROCM = os.environ.get('ROCM_PATH', '/shared/apps/ubuntu/opt/rocm-7.0.2')
CLANG = os.environ.get('CLANG', '/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang')

# (family label, dispatch kind or None, ours host family, aiter dir, aiter host family, shapes)
FAMS = [
    ('MXFP4 (a4w4)', None, 'ours_v4_f4', 'f4gemm', 'asm_f4', [(8192, 8192, 8192), (4096, 4096, 4096)]),
    ('int8 per-token', 'i8_rowcol', 'ours_v4_i8', 'i8gemm', 'asm_i8',
     [(8192, 8192, 8192), (4096, 4096, 4096), (1024, 4096, 8192), (256, 7168, 4096), (128, 4096, 8192)]),
    ('fp8 blockscale', 'fp8_blockscale', 'ours_v4_fp8bs', 'fp8gemm_blockscale', 'asm_fp8bs',
     [(4096, 4096, 4096), (1024, 4096, 8192), (512, 7168, 7168), (256, 7168, 4096), (128, 4096, 8192)]),
    ('bf16', 'bf16', 'ours_v4_bf16', 'bf16gemm', 'asm_bf16_pre', [(2048, 4096, 4096), (128, 4096, 8192)]),
]
# v9.2 ratios (docs/op-coverage-design.md section 7.4, 2026-11-09)
V92 = {('MXFP4 (a4w4)', (8192, 8192, 8192)): 0.90, ('MXFP4 (a4w4)', (4096, 4096, 4096)): 0.86,
       ('int8 per-token', (8192, 8192, 8192)): 1.53, ('int8 per-token', (4096, 4096, 4096)): 1.63, ('int8 per-token', (1024, 4096, 8192)): 1.08,
       ('int8 per-token', (256, 7168, 4096)): 1.01, ('int8 per-token', (128, 4096, 8192)): 1.27,
       ('fp8 blockscale', (4096, 4096, 4096)): 1.05, ('fp8 blockscale', (1024, 4096, 8192)): 0.98, ('fp8 blockscale', (512, 7168, 7168)): 1.01,
       ('fp8 blockscale', (256, 7168, 4096)): 0.91, ('fp8 blockscale', (128, 4096, 8192)): 1.45,
       ('bf16', (2048, 4096, 4096)): 0.96, ('bf16', (128, 4096, 8192)): None}


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
    """Median us over `reps` runs; (us, ok, line)."""
    ts, ok, last = [], True, ''
    for _ in range(reps):
        rc, out = sh([host] + [str(a) for a in args], env)
        m = re.search(r': ([0-9.]+) us .* bad=(\d+)', out)
        if rc != 0 or not m:
            return None, False, out.strip().splitlines()[-1] if out.strip() else f'rc={rc}'
        ts.append(float(m.group(1)))
        ok &= int(m.group(2)) == 0
        last = [l for l in out.splitlines() if ' us ' in l][-1]
    return statistics.median(ts), ok, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amdc', default='/tmp/target-kfmha/release/amdc')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default='build/vs_aiter_gemm')
    ap.add_argument('--only', default='', help='substring of the family label')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    host = os.path.join(a.out, 'vendor_gemm_host')
    rc, out = sh([os.path.join(ROCM, 'bin', 'amdclang++'), '-x', 'hip', '-O2', '--offload-arch=gfx950', '--rocm-path=' + ROCM, '-std=c++17',
                  '-I', os.path.join(ROOT, 'tests', 'gpu'), os.path.join(ROOT, 'tests', 'gpu', 'vendor_gemm_host.cpp'), '-o', host,
                  '-L' + os.path.join(ROCM, 'lib'), '-lamdhip64', '-Wl,-rpath,' + os.path.join(ROCM, 'lib')])
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

    rows = []
    for label, kind, ofam, vdir, vfam, shapes in FAMS:
        if a.only and a.only not in label:
            continue
        vcos = sorted(f for f in os.listdir(os.path.join(HSA, vdir)) if f.endswith('.co'))
        for (m, n, k) in shapes:
            # ours: the dispatched kernel
            if kind:
                rc, out = sh([a.amdc, 'dispatch', 'gemm', kind, str(m), str(n), str(k)])
                p = json.loads(out[out.index('{'):])
                kn, s, red = p['kernel'], p['splits'], p.get('reduce')
            else:
                kn, s, red = 'gemm_a4w4_mxfp4', 1, None
            env = dict(SPLITK=s, ITERS=100, WARMUP=20, CHECK_ROWS=8)
            if red:
                env.update(REDUCE_CO=co_of(red), REDUCE_KN=red)
            tm, tn = our_tile(kn)
            ou, ook, oline = run_host(host, [ofam, co_of(kn), kn, m, n, k, tm, tn], env, a.reps)
            # aiter: the fastest code object of the family
            best = None
            for f in vcos:
                t = tile_of_name(f)
                if not t:
                    continue
                if vfam == 'asm_bf16_pre' and 'bpreshuffle' not in f:
                    continue
                if vfam == 'asm_bf16_pre' and 'splitk' in f:
                    continue
                path = os.path.join(HSA, vdir, f)
                vu, vok, vline = run_host(host, [vfam, path, vname(path), m, n, k, t[0], t[1]], dict(ITERS=100, WARMUP=20, CHECK_ROWS=8), a.reps)
                if vu is not None and vok and (best is None or vu < best[0]):
                    best = (vu, f.replace('.co', ''))
            r = (best[0] / ou) if (ou and best) else None
            rows.append(dict(family=label, shape=f'{m}x{n}x{k}', ours_us=ou, ours_ok=ook, ours_kernel=kn + (f' S={s}' if s > 1 else ''),
                             aiter_us=best[0] if best else None, aiter_kernel=best[1] if best else None, ratio=r, v92=V92.get((label, (m, n, k)))))
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(a.out, 'gemm_vs_aiter.json'), 'w'), indent=1)
    print('\n| family | shape | ours (us, kernel) | aiter (us, best tile) | ratio | v9.2 |')
    print('|---|---|--:|--:|--:|--:|')
    for r in rows:
        f = lambda x, d=1: '-' if x is None else f'{x:.{d}f}'
        print(f"| {r['family']} | {r['shape']} | {f(r['ours_us'])} ({r['ours_kernel']}{'' if r['ours_ok'] else ', BAD'}) | {f(r['aiter_us'])} ({r['aiter_kernel']}) | {f(r['ratio'], 2)} | {f(r['v92'], 2)} |")


if __name__ == '__main__':
    main()
