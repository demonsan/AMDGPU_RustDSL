#!/usr/bin/env python3
"""End-to-end JIT benchmark (docs/compile-time.md): Python `get_kernel` -> amdc (compile + in-process assemble + info) -> disk cache -> HIP module load.

    AMDGPU_JIT_AMDC=/path/amdc python3 tools/jit_bench.py [--kernels a,b,..] [--n 32] [--workers 1,4,16] [--cache /tmp/jb-cache]

Each kernel is specialized on its first specializable 32-bit kernarg (n distinct values = n distinct shapes). Phases: cold sequential (empty disk cache), disk-warm (fresh process state, populated
cache: key + read + hipModuleLoadData), memory-warm, cold parallel (ThreadPoolExecutor with W workers over all shapes). Needs a GPU for the module load (HIP_VISIBLE_DEVICES)."""
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

ap = argparse.ArgumentParser()
ap.add_argument('--kernels', default='silu_mul_bf16_d4096,rmsnorm_quant_fp8_n4096,gemm_a8w8_rowcol_fp8_64,pa_decode16_bf16_hd128_p16_f')
ap.add_argument('--n', type=int, default=32)
ap.add_argument('--workers', default='1,4,16')
ap.add_argument('--cache', default=os.path.join(tempfile.gettempdir(), 'jit-bench-cache'))
a = ap.parse_args()
amdc = os.environ.get('AMDGPU_JIT_AMDC')
if not amdc:
    sys.exit('set AMDGPU_JIT_AMDC')
os.environ['AMDGPU_JIT_CACHE_DIR'] = a.cache

from amdgpu_jit import _compile as C  # noqa: E402


def fresh():
    shutil.rmtree(a.cache, ignore_errors=True)
    C._modules.clear()
    C._fast.clear()


def forget():
    C._modules.clear()
    C._fast.clear()


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
    specs += [(k, {nm: 64 * (i + 1)}) for i in range(a.n)]
print(f'{len(specs)} shapes ({len(kernels)} kernels x {a.n}), amdc = {amdc}, cpu = {os.cpu_count()}\n')

rows = []
# cold sequential
fresh()
per = {}
child = []
orig = C._compile_into


def timed(*args, **kw):
    t = time.perf_counter()
    try:
        return orig(*args, **kw)
    finally:
        child.append((time.perf_counter() - t) * 1e3)


C._compile_into = timed
t0 = time.perf_counter()
for k, sp in specs:
    t = time.perf_counter()
    C.get_kernel(k, sp)
    per.setdefault(k, []).append((time.perf_counter() - t) * 1e3)
tot = time.perf_counter() - t0
C._compile_into = orig
print('## cold, sequential (empty disk cache): ms per kernel = Python + amdc subprocess + cache write + HIP module load')
print('| kernel | median | p10 | p90 |\n|---|--:|--:|--:|')
for k in kernels:
    if k in per:
        m, lo, hi = stats_ms(per[k])
        print(f'| {k} | {m:.1f} | {lo:.1f} | {hi:.1f} |')
print(f'\nall {len(specs)} shapes: {tot:.2f} s = {tot * 1e3 / len(specs):.1f} ms per kernel; of which the compile subprocess phase (`_compile_into`) {statistics.mean(child):.1f} ms mean\n')

# disk-warm
forget()
per2 = {}
t0 = time.perf_counter()
for k, sp in specs:
    t = time.perf_counter()
    C.get_kernel(k, sp)
    per2.setdefault(k, []).append((time.perf_counter() - t) * 1e3)
tot2 = time.perf_counter() - t0
print('## disk-warm (new process state, populated cache): key + read + module load')
print('| kernel | median | p10 | p90 |\n|---|--:|--:|--:|')
for k in kernels:
    if k in per2:
        m, lo, hi = stats_ms(per2[k])
        print(f'| {k} | {m:.2f} | {lo:.2f} | {hi:.2f} |')
print(f'\nall: {tot2 * 1e3 / len(specs):.2f} ms per kernel\n')

# memory-warm
t0 = time.perf_counter()
reps = 20
for _ in range(reps):
    for k, sp in specs:
        C.get_kernel(k, sp)
tot3 = time.perf_counter() - t0
print(f'## memory-warm: {tot3 * 1e6 / (reps * len(specs)):.1f} us per get_kernel\n')

# cold parallel
print('## cold, parallel over all shapes (fresh disk cache each)')
print('| workers | wall (s) | ms per kernel | speed-up vs sequential |\n|--:|--:|--:|--:|')
for w in [int(x) for x in a.workers.split(',')]:
    fresh()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(lambda ks: C.get_kernel(ks[0], ks[1]), specs))
    wall = time.perf_counter() - t0
    print(f'| {w} | {wall:.2f} | {wall * 1e3 / len(specs):.1f} | {tot / wall:.1f}x |')

# the subprocess floor
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