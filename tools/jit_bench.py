#!/usr/bin/env python3
"""End-to-end JIT benchmark (docs/compile-time.md): Python `get_kernel` -> amdc (compile + certificate + in-process assemble + info) -> disk cache -> HIP module load.

    AMDGPU_JIT_AMDC=/path/amdc python3 tools/jit_bench.py [--kernels a,b,..] [--n 32] [--workers 1,4,16,32] [--cache /tmp/jb-cache]

Each kernel is specialized on its first specializable 32-bit kernarg (n distinct values = n distinct shapes). Three configurations of the same cold work:
  1. a fresh `amdc emit` process per kernel (AMDGPU_JIT_SERVER=0),
  2. the persistent compile server (`amdc serve`),
  3. the server with the certificates pre-populated offline (`amdc certify-batch`).
For each: cold sequential per kernel and cold parallel over all shapes (ThreadPoolExecutor); then disk-warm (fresh process state, populated cache) and memory-warm. Needs a GPU for the module load."""
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

ap = argparse.ArgumentParser()
ap.add_argument('--kernels', default='silu_mul_bf16_d4096,rmsnorm_quant_fp8_n4096,gemm_a8w8_rowcol_fp8_64,pa_decode16_bf16_hd128_p16_f')
ap.add_argument('--n', type=int, default=32)
ap.add_argument('--workers', default='1,4,16,32')
ap.add_argument('--cache', default=os.path.join(tempfile.gettempdir(), 'jit-bench-cache'))
a = ap.parse_args()
amdc = os.environ.get('AMDGPU_JIT_AMDC')
if not amdc:
    sys.exit('set AMDGPU_JIT_AMDC')
os.environ['AMDGPU_JIT_CACHE_DIR'] = a.cache

from amdgpu_jit import _compile as C  # noqa: E402


def forget():
    C._modules.clear()
    C._fast.clear()


def fresh():
    shutil.rmtree(a.cache, ignore_errors=True)
    forget()


def spec_arg(k):
    j = json.loads(subprocess.run([amdc, 'jit-info', k, '--backend', 'own'], capture_output=True, text=True, check=True).stdout)
    for x in j['args']:
        if x.get('specializable') and x.get('size') == 4 and x.get('kind') in ('i32', 'u32'):
            return x['name']
    return None


def stats_ms(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return statistics.median(v), q(0.1), q(0.9)


kernels = a.kernels.split(',')
specs = []
for k in kernels:
    nm = spec_arg(k)
    if nm is None:
        print('skip', k, '(no specializable i32 argument)')
        continue
    specs += [(k, {nm: 4 * (i + 1)}) for i in range(a.n)]
workers = [int(x) for x in a.workers.split(',')]
print(f'{len(specs)} shapes ({len(kernels)} kernels x {a.n}), amdc = {amdc}, cpu = {os.cpu_count()}\n')


def precertify():
    f = os.path.join(tempfile.gettempdir(), 'jit-bench-shapes.tsv')
    with open(f, 'w') as fh:
        for k, sp in specs:
            fh.write(k + '\t' + C.spec_string(sp) + '\n')
    t = time.perf_counter()
    r = subprocess.run([amdc, 'certify-batch', f, '--cert-cache', C.cert_cache_dir(amdc), '--jobs', str(os.cpu_count() or 8)], capture_output=True, text=True, check=True)
    return time.perf_counter() - t, r.stdout.strip()


configs = [('1. one `amdc emit` process per kernel', {'AMDGPU_JIT_SERVER': '0'}, False),
           ('2. persistent compile server', {'AMDGPU_JIT_SERVER': '1'}, False),
           ('3. persistent server + certificates pre-populated (`amdc certify-batch`)', {'AMDGPU_JIT_SERVER': '1'}, True)]
summary = []
for label, env, pre in configs:
    for kk, v in env.items():
        os.environ[kk] = v
    C.close_servers()
    fresh()
    note = ''
    if pre:
        dt, out = precertify()
        note = f' (offline certification of {len(specs)} shapes: {dt:.2f} s)'
    print(f'## {label}{note}\n')
    per = {}
    t0 = time.perf_counter()
    for k, sp in specs:
        t = time.perf_counter()
        C.get_kernel(k, sp)
        per.setdefault(k, []).append((time.perf_counter() - t) * 1e3)
    tot = time.perf_counter() - t0
    print('| kernel | cold sequential: median ms | p10 | p90 |\n|---|--:|--:|--:|')
    for k in kernels:
        if k in per:
            m, lo, hi = stats_ms(per[k])
            print(f'| {k} | {m:.1f} | {lo:.1f} | {hi:.1f} |')
    print(f'\nall {len(specs)} shapes cold, sequential: {tot:.2f} s = **{tot * 1e3 / len(specs):.1f} ms per kernel**\n')
    row = [f'{tot * 1e3 / len(specs):.1f}']
    print('| workers | wall (s) | ms per kernel |\n|--:|--:|--:|')
    for w in workers:
        C.close_servers()
        fresh()
        if pre:
            precertify()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(lambda ks: C.get_kernel(ks[0], ks[1]), specs))
        wall = time.perf_counter() - t0
        print(f'| {w} | {wall:.2f} | {wall * 1e3 / len(specs):.2f} |')
        row.append(f'{wall * 1e3 / len(specs):.2f}')
    print()
    summary.append((label, row))
    if pre:
        forget()
        t0 = time.perf_counter()
        for k, sp in specs:
            C.get_kernel(k, sp)
        d = time.perf_counter() - t0
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            for k, sp in specs:
                C.get_kernel(k, sp)
        m = time.perf_counter() - t0
        print(f'disk-warm (new process state, populated cache): **{d * 1e3 / len(specs):.2f} ms per kernel**; memory-warm: **{m * 1e6 / (reps * len(specs)):.1f} us per call**\n')

print('## summary: ms per kernel (cold)\n')
print('| configuration | sequential | ' + ' | '.join(f'{w} workers' for w in workers) + ' |')
print('|---|--:|' + '--:|' * len(workers))
for label, row in summary:
    print(f'| {label} | ' + ' | '.join(row) + ' |')

d = tempfile.mkdtemp()
k = kernels[0]
nm = spec_arg(k)
ts = []
for _ in range(10):
    t = time.perf_counter()
    subprocess.run([amdc, 'emit', k, '--backend', 'own', '--out', os.path.join(d, 'x.s'), '--co', os.path.join(d, 'x.co'), '--info', os.path.join(d, 'i.json'), '--specialize', f'{nm}=64'], capture_output=True, check=True)
    ts.append((time.perf_counter() - t) * 1e3)
print(f'\nthe amdc process alone (`emit --co --info`, {k}): median {statistics.median(ts):.1f} ms wall')
shutil.rmtree(d, ignore_errors=True)
C.close_servers()
