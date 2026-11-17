"""The case library of tools/perf_ab.py for the MI450 (gfx1250) nodes: `perf_ab.py --arch gfx1250` (docs/mi450-nodes.md, docs/perf-ab.md).

Same contract as tools/perf_ab_cases.py (a case = what one commit executes + how the result is read + an optional vendor side), with the gfx1250 conventions:
- every GPU run goes through the PERF_AB_GPU_WRAP prefix (`timeout -s KILL` and, on the lock node, gpu-lock), the GPU is the idle one the runner picked (HIP_VISIBLE_DEVICES);
- the hosts print ONE JSON line per run after their human line: {"host":..,"kernel":..,"shape":{..},"us":..,"tflops"|"gbs":..,"bad":N,"ok":bool}; the metrics are `json_us` / `json_tflops` / `json_gbs`;
- hang safety: every host polls its waits against a deadline and leaves with exit 124 WITHOUT touching the device again; a case never retries (a non-zero exit is a FAIL of the case, reported, not repeated);
- the vendor side is runtime-only content: the code objects live in $AITER_HSA (default $BASE/vendor/gfx1250 on the node), never in the repo; a case whose vendor code object is missing simply has no vendor side.

Add a case by appending to `cases()`: ours = `Run(emit=[kernel names for `amdc emit --target gfx1250`], cmd=[{host:..}, {co:..}, ...], env)`, vendor = a callable returning a `Run` without emit.
"""
import os
import re

from perf_ab import Case, Run, HSA, ROOT, ROCM, CLANG, sh, ARCH


def _vname(path):
    """Kernel symbol of a vendor code object (the first mangled kernel name in its strings)."""
    try:
        data = open(path, 'rb').read()
    except OSError:
        return 'MISSING'
    m = re.search(rb'_ZN[0-9A-Za-z_]+E', data)
    return m.group(0).decode() if m else 'MISSING'


def _prebuilt_smoke(tree):
    """The smoke kernel built by the node's ROCm clang (until our own emitter targets the architecture): the harness bring-up case."""
    out = os.path.join(ROOT, 'build', 'ab', ARCH)
    co = os.path.join(out, 'smoke.co')
    src = os.path.join(tree, 'tests', 'gpu', 'smoke_kernel.hip')
    if not os.path.exists(src):
        return None
    if not os.path.exists(co) or os.path.getmtime(co) < os.path.getmtime(src):
        os.makedirs(out, exist_ok=True)
        p = sh([os.path.join(ROCM, 'bin', 'amdclang++'), '-x', 'hip', f'--offload-arch={ARCH}', '--cuda-device-only', '-O1', '-o', co, src])
        if p.returncode != 0:
            return None
    return co


def smoke(tree):
    co = _prebuilt_smoke(tree)
    if co is None:
        return []
    out = []
    for n in (1 << 20, 1 << 24):
        def build(side, n=n):
            return Run([], ['{host:smoke_host}', co, 'smoke_iota', str(n), '256'], dict(ITERS=200))
        out.append(Case(id=f'smoke/iota/n{n}', family='smoke', build=build, metric=['json_gbs', 'json_us'], line=r'^\{"host":"smoke"', ok=r'SMOKE_OK', timeout=120))
    return out


def f4gemm(tree):
    """aiter f4gemm (MXFP4 GEMM) through the cluster-aware host (tests/gpu/g1250_f4gemm_host.cpp): the vendor baseline until our own replica exists; with no ours-side yet the case measures
    the vendor kernel on the `new` side (the old/new pair is then the noise floor of the harness)."""
    host = os.path.join(tree, 'tests', 'gpu', 'g1250_f4gemm_host.cpp')
    co = os.path.join(HSA, 'f4gemm', 'f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_ps.co')
    if not (os.path.exists(host) and os.path.exists(co)):
        return []
    out = []
    for (m, n, k) in [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
        def build(side, m=m, n=n, k=k):
            return Run([], ['{host:g1250_f4gemm_host}', co, _vname(co), str(m), str(n), str(k), '1', '50', '20000'], {})
        out.append(Case(id=f'f4gemm/vendor/{m}x{n}x{k}', family='f4gemm', build=build, metric=['json_us', 'json_tflops'], line=r'^\{"host"', ok=r'"ok":\s*true', timeout=300))
    return out


def mla(tree):
    """aiter MLA decode (fp8 Q / KV, qh16, page 64, one split) through tests/gpu/g1250_mla_host.cpp: the vendor baseline until our own replica exists (the old/new pair is then the noise floor).
    The host checks against an fp32 CPU reference (aiter's tolerance plus a cosine) and prints one JSON line."""
    host = os.path.join(tree, 'tests', 'gpu', 'g1250_mla_host.cpp')
    co = os.path.join(HSA, 'mla', 'mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co')
    if not (os.path.exists(host) and os.path.exists(co)):
        return []
    out = []
    # hot: the KV stays in L2 / MALL between launches (the decode-step working set of a small batch); flushed: every launch reads the KV from HBM (FLUSH=1, tests/gpu/g1250_timing.h)
    for flush in (0, 1):
        for (b, ctx) in [(1, 64), (4, 1000), (32, 2048), (128, 4096)]:
            def build(side, b=b, ctx=ctx, flush=flush):
                return Run([], ['{host:g1250_mla_host}', co, _vname(co), str(b), str(ctx), '16', '50', '20000'], dict(FLUSH=flush))
            out.append(Case(id=f'mla/vendor/{"flushed" if flush else "hot"}/B{b}C{ctx}', family='mla', build=build, metric=['json_us', 'json_gbs'], line=r'^\{"host"', ok=r'"ok":\s*true', timeout=300))
    return out


def cases(tree, args):
    out = []
    for f in (smoke, f4gemm, mla, paired_floor):
        out += f(tree)
    # the --quick set: the first and the last case of every family
    groups = {}
    for c in out:
        groups.setdefault(c.family, []).append(c)
    for cs in groups.values():
        cs[0].quick = True
        cs[-1].quick = True
    return out


def paired_floor(tree):
    """Harness validation: the SAME vendor code object as old and new, timed paired (alternating blocks on the same buffers in one process). The ratio must be 1.000 within the pair spread whatever the
    placement of the buffers: this is the noise floor of a paired A/B (the unpaired old / new pair of `mla/vendor/*` varies by tens of percent between processes)."""
    out = []
    mhost = os.path.join(tree, 'tests', 'gpu', 'g1250_mla_host.cpp')
    mco = os.path.join(HSA, 'mla', 'mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co')
    if os.path.exists(mhost) and os.path.exists(mco):
        for (b, ctx) in [(1, 64), (4, 1000), (32, 2048), (128, 4096)]:
            def build(side, b=b, ctx=ctx):
                import shutil
                os.makedirs(side.codir, exist_ok=True)
                shutil.copy(mco, os.path.join(side.codir, 'mlaself.co'))   # a byte-identical copy per side: two loads of the same code at different addresses
                return Run([], ['{host:g1250_mla_host}', '{oldco:mlaself}', _vname(mco), str(b), str(ctx), '16', '50', '20000'], dict(CO2='{co:mlaself}'))
            out.append(Case(id=f'mla/paired-self/B{b}C{ctx}', family='paired', build=build, metric='json_us', line=r'"paired"', ok=r'"ok":\s*true', paired=True, timeout=300))
    return out
