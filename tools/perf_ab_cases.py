"""The case library of tools/perf_ab.py (docs/perf-ab.md): every integrated kernel family, each case = what one commit executes (kernels to emit, host command) + how the
result is read (metric, correctness marker) + an optional vendor side.

A case's `build(side)` may ask the commit's own launch rules (`side.dispatch("prefill_bf16", ...)` = `amdc dispatch ...`), so a commit that changes a rule is measured with ITS rule.
The command tokens: `{co:NAME}` = the assembled kernel `NAME` of that commit, `{codir}` = the dir of all its kernels, `{host:NAME}` = the host binary built from tests/gpu/NAME.cpp.
Add a case by appending to one of the family functions below; keep shapes few and representative (the whole table runs in one session, a few minutes per repetition).
"""
import math
import os
import re
import shlex

from perf_ab import Case, Run, HSA, REF_CO, ROOT


def log2(n):
    return int(round(math.log2(n)))


def _vname(path):
    """Kernel symbol of an aiter vendor code object (the first mangled kernel name in its strings)."""
    try:
        data = open(path, 'rb').read()
    except OSError:
        return 'MISSING'
    m = re.search(rb'_ZN[0-9A-Za-z_]+E', data)
    return m.group(0).decode() if m else 'MISSING'


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# memory-bound families: every timed case of tools/b2_matrix.sh (parsed, so the matrix stays the single list)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def membound(tree):
    path = os.path.join(tree, 'tools', 'b2_matrix.sh')
    if not os.path.exists(path):
        return []
    text = open(path, encoding='utf-8').read()
    hosts = dict(re.findall(r'\b([A-Z][A-Z0-9]*)=\$O/([a-z0-9_]+_host)\b', text))
    out = []
    for line in text.splitlines():
        m = re.match(r'^((?:[A-Z_]+=\S+ )*)one "([^"]*)" (.*)$', line)
        if not m:
            continue
        envs, label, rest = m.groups()
        if re.search(r'\bcorr\b|correct', label):
            continue  # correctness-only shapes (a few rows): no bandwidth to compare
        toks = shlex.split(rest)
        cmd, emit = [], []
        for t in toks:
            v = re.fullmatch(r'\$([A-Z][A-Z0-9]*)', t)
            c = re.fullmatch(r'\$O/([A-Za-z0-9_]+)\.co', t)
            if v and v.group(1) in hosts:
                cmd.append('{host:%s}' % hosts[v.group(1)])
            elif c:
                cmd.append('{co:%s}' % c.group(1))
                emit.append(c.group(1))
            else:
                cmd.append(t)
        if not cmd or not cmd[0].startswith('{host:'):
            continue
        env = dict(e.split('=', 1) for e in envs.split())
        out.append(Case(id=f'mb/{label}', family='membound', build=(lambda side, cmd=cmd, emit=emit, env=env: Run(emit, cmd, env)),
                        metric=['gbs', 'us'], line=r'^timing', ok=r'^check.*OK|bad[a-z_]*=0', timeout=300))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# attention prefill / backward
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def _vendor_attn(co_name, shape, causal, extra_args=(), env=None):
    v = os.path.join(HSA, 'fmha_v3_fwd', co_name)

    def f():
        return Run([], ['{host:attn_host}', 'vendor', v, _vname(v)] + [str(x) for x in shape] + [str(causal)] + [str(x) for x in extra_args], dict(env or {}))
    return f


def attn_bf16():
    out = []
    shapes = ['2 32 32 8192 8192', '4 16 16 2048 2048', '1 8 8 4096 4096', '4 16 16 1024 1024', '1 32 8 2048 2048']
    for c in (0, 1):
        for sh in shapes:
            shp = [int(x) for x in sh.split()]

            def build(side, shp=shp, c=c):
                p = side.dispatch('prefill_bf16', *shp[:3], shp[4], c) if False else side.dispatch('prefill_bf16', shp[0], shp[1], shp[3], shp[4], c)
                k, s = p['kernel'], p['splits']
                rows = 128 if (k.endswith('_q128') or k == 'fmha_fwd_hd128_bf16_causal') else 256
                pair = 0 if k == 'fmha_fwd_hd128_bf16_causal' else c
                emit, env = [k], dict(LSE=0, CHECK=4, ROWS=rows, BLOCK=rows * 2, PAIR=pair)
                if s > 1:
                    env.update(KS=log2(s), COMBINE_CO='{co:attn_combine_bf16_hd128_q16}')
                    emit.append('attn_combine_bf16_hd128_q16')
                return Run(emit, ['{host:attn_host}', 'ours', '{co:%s}' % k, k] + [str(x) for x in shp] + [str(c)], env)
            cid = f'bf16/{"causal" if c else "full"}/' + 'x'.join(sh.split())
            co = 'fwd_hd128_bf16%s.co' % ('_causal' if c else '')
            out.append(Case(id=cid, family='fmha_bf16', build=build, metric='tflops', line='TF/s', ok='ATTN_OK',
                            vendor=_vendor_attn(co, shp, c, env=dict(LSE=0, CHECK=4)), timeout=900))
    return out


def attn_hd192():
    out = []
    for c in (0, 1):
        for sh in ['1 16 16 4096 4096', '4 16 16 2048 2048', '1 8 8 4096 4096', '1 128 128 1024 1024']:
            shp = [int(x) for x in sh.split()]

            def build(side, shp=shp, c=c):
                p = side.dispatch('hd192', shp[0], shp[1], shp[3], shp[4], c)
                k, s = p['kernel'], p['splits']
                rows = 128 if '_q128' in k else 256
                pair = 1 if (c and not ('_np' in k or '_ks' in k)) else 0
                emit, env = [k], dict(DV=128, LSE=0, CHECK=4, ROWS=rows, BLOCK=rows * 2, PAIR=pair)
                if s > 1:
                    env.update(KS=log2(s), COMBINE_CO='{co:attn_combine_bf16_hd128_q16}', COMBINE_KN='attn_combine_bf16_hd128_q16')
                    emit.append('attn_combine_bf16_hd128_q16')
                return Run(emit, ['{host:attn_host}', 'ours', '{co:%s}' % k, k] + [str(x) for x in shp] + [str(c), '192'], env)
            co = 'fwd_hd192_hd128_bf16%s.co' % ('_causal' if c else '')
            out.append(Case(id=f'hd192/{"causal" if c else "full"}/' + 'x'.join(sh.split()), family='fmha_hd192', build=build, metric='tflops', line='TF/s', ok='ATTN_OK',
                            vendor=_vendor_attn(co, shp, c, ['192'], dict(ROWS=128, BLOCK=256, DV=128, LSE=0, CHECK=4)), timeout=900))
    return out


def attn_fp8pt():
    out = []
    for c in (0, 1):
        for sh in ['2 32 32 8192 8192', '4 16 16 2048 2048', '1 8 8 4096 4096', '2 32 8 4096 4096']:
            shp = [int(x) for x in sh.split()]
            s_ = '_causal' if c else ''
            vco = os.path.join(HSA, 'fmha_v3_fwd', f'fwd_hd128_fp8{s_}.co')

            def build(side, shp=shp, c=c, vco=vco):
                p = side.dispatch('fp8pt', shp[0], shp[1], shp[3], shp[4], c)
                k, s = p['kernel'], p['splits']
                emit, env = [k], dict(SCALAR_DS=1, CHECK=16)
                if s > 1:
                    env.update(KVS=s, COMBINE_CO='{co:fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h}')
                    emit.append('fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h')
                return Run(emit, ['{host:fp8pt_host}', vco, _vname(vco), '{co:%s}' % k, k] + [str(x) for x in shp] + [str(c)], env)
            out.append(Case(id=f'fp8pt/{"causal" if c else "full"}/' + 'x'.join(sh.split()), family='fmha_fp8pt', build=build, metric='tflops', line=r'^ours B=.*TF/s', ok=r'FP8PT_OK',
                            vendor=True, vendor_same=True, vendor_line=r'^vendor B=.*TF/s', timeout=900))
    return out


def attn_mxfp8():
    """The _vs128 kernel forced at 256 rows per block (as tools/attn_family_table.sh; the KV-split / t128 paths are covered by tools/port_gate.sh)."""
    out = []
    k = 'fmha_fwd_hd128_mxfp8_v3_vs128'
    for sh in ['1 8 8192', '4 8 2048', '2 32 8192', '1 8 2048']:
        b, h, s = [int(x) for x in sh.split()]

        def build(side, b=b, h=h, s=s):
            env = dict(REF_B=b, REF_H=h, REF_S=s, REF_SK=s, REF_DATA='normal', REF_ROWS=4, REF_WARMUP=50, REF_ITERS=50, REF_CO=REF_CO,
                       REF_CO2='{co:%s}' % k, REF_KNAME2=k, REF_ROWS_PER_BLOCK=256)
            return Run([k], ['{host:ref_host}', REF_CO], env)
        out.append(Case(id='mxfp8/' + 'x'.join(sh.split()), family='fmha_mxfp8', build=build, metric='tflops', line=r'^timing ours', ok='REF_KERNEL_OK', vendor=True, vendor_same=True, vendor_line=r'^timing vendor', timeout=900))
    return out


def attn_bwd():
    out = []
    for c in (0, 1):
        for sh in ['1 16 16 1024 1024', '1 16 4 1024 1024', '2 8 8 1024 1024']:
            shp = sh.split()

            def build(side, shp=shp, c=c):
                if c:
                    p = side.dispatch('bwd_causal')
                    dq, dkdv = p['dq'], p['dkdv']
                else:
                    dq, dkdv = 'fmha_bwd_dq_hd128_bf16', 'fmha_bwd_dkdv_hd128_bf16'
                emit = ['fmha_bwd_dot_hd128_bf16', dq, dkdv]
                return Run(emit, ['{host:attn_bwd_host}', '{codir}'] + shp + [str(c)], dict(VENDOR_DIR=os.path.join(HSA, 'fmha_v3_bwd'), CHECK=4))
            out.append(Case(id=f'bwd/{"causal" if c else "full"}/' + 'x'.join(shp), family='fmha_bwd', build=build, metric='tflops', line=r'^ours bwd .*TF/s', ok=r'BWD_OK', vendor=True, vendor_same=True, vendor_line=r'^vendor bwd .*TF/s', timeout=1200))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# decode: paged (pa_host), MLA (mla_host)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def decode():
    out = []
    for (b, h, hk, ctx) in [(1, 64, 8, 8192), (8, 64, 8, 4096), (32, 64, 8, 2048), (64, 32, 4, 1024), (4, 16, 1, 16384), (1, 32, 4, 32768)]:
        def build(side, b=b, h=h, hk=hk, ctx=ctx):
            page = 16
            p = side.dispatch('decode16', b, hk, ctx, '--page', page)
            k, form, s = p['kernel'], p['form'], p['splits']
            env = dict(KPREFIX='pa_decode16_bf16_hd128_p')
            env.update(dict(FUSED=1, KSUFFIX='_f') if form == 'fused' else dict(LASTC=1, KSUFFIX='_c'))
            return Run([k, 'attn_combine_bf16_hd128'], ['{host:pa_host}', 'ours', '{co:%s}' % k, '{co:attn_combine_bf16_hd128}', str(b), str(h), str(hk), str(ctx), str(page), str(s)], env)
        out.append(Case(id=f'decode16/B{b}H{h}HK{hk}C{ctx}', family='decode', build=build, metric='us', line=r'^ours ', ok=r'PA_OK', timeout=600))
    return out


def mla():
    out = []
    for (b, h, ctx) in [(1, 128, 8192), (8, 128, 4096), (32, 16, 2048), (4, 16, 16384), (1, 16, 2048), (4, 16, 4096), (16, 16, 1024), (1, 32, 2048), (4, 16, 8192), (64, 16, 4096), (2, 16, 3072), (8, 32, 4096), (12, 48, 2048)]:
        def build(side, b=b, h=h, ctx=ctx):
            p = side.dispatch('mla', b, h, '--ctx', ctx)
            hp, s = p['hp'], p['splits']
            page = 64
            k = f'mla_decode_bf16_{"hp_" if hp else ""}p{page}'
            return Run([k, 'attn_combine_bf16_hd512'], ['{host:mla_host}', '{co:%s}' % k, '{co:attn_combine_bf16_hd512}', str(b), str(h), str(ctx), str(page), str(s)], dict(MLAHP=1 if hp else 0))
        out.append(Case(id=f'mla/B{b}H{h}C{ctx}', family='mla', build=build, metric=['us', 'gbs'], line=r'us|GB/s', ok=r'OK|PASS', timeout=600))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# MoE stages (moe_host: stage 1 + stage 2 + sum; moe_fp4_host), conv
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def moe():
    out = []
    parts = [dict(name='stage1', metric='us', line=r'stage1', vendor_line=None, better=None), dict(name='stage2', metric=r'(?:stage2\+sum|requant\+stage2\+sum) ([0-9.]+) us', line=r'stage', vendor_line=None, better='low')]
    parts[0]['metric'] = r'stage1 ([0-9.]+) us'
    parts[0]['better'] = 'low'
    for dt, (m, d, i, e, topk) in [('bf16', (4096, 2048, 768, 64, 8)), ('bf16', (256, 2048, 768, 64, 8)), ('fp8', (2048, 4096, 256, 32, 8)), ('fp8', (512, 4096, 256, 32, 8)),
                                   # the block_m rule change (E16 / E32 band, measured) and shapes it was not fitted on (other D / I, E) + unchanged controls
                                   ('fp8', (640, 4096, 256, 32, 8)), ('bf16', (320, 2048, 768, 16, 8)), ('bf16', (384, 2048, 768, 16, 8)), ('fp8', (1280, 4096, 256, 64, 8)), ('bf16', (1024, 2048, 768, 32, 8))]:
        def build(side, dt=dt, m=m, d=d, i=i, e=e, topk=topk):
            bm = side.dispatch('moe', m, topk, e, '--inter', i)['block_m']
            bm = bm if bm in (64, 128) else 128
            sfx = '_bm64' if bm == 64 else ''
            ks = [f'gemm_moe1_{dt}_silu{sfx}', f'gemm_moe2_{dt}{sfx}', f'moe_sum_bf16_k{topk}']
            if dt == 'fp8':
                ks.append(f'quant_token_bf16_fp8_n{i}')   # the host re-quantizes the stage-1 output between the stages with it
            return Run(ks, ['{host:moe_host}', dt, '{codir}', str(m), str(d), str(i), str(e), str(topk), 'silu'], dict(BLOCK_M=bm, CHECK=2))
        out.append(Case(id=f'moe/{dt}/M{m}D{d}I{i}E{e}K{topk}', family='moe', build=build, metric='us', parts=parts, ok=r'MOE_OK', timeout=900))
    for (m, d, i, e, topk) in [(2048, 7168, 256, 64, 8), (256, 7168, 256, 64, 8), (512, 7168, 256, 32, 8), (384, 7168, 256, 16, 8)]:
        def build(side, m=m, d=d, i=i, e=e, topk=topk):
            bm = side.dispatch('moe', m, topk, e, '--inter', i)['block_m']
            bm = bm if bm in (64, 128) else 128
            sfx = '_bm64' if bm == 64 else ''
            ks = [f'gemm_moe1_fp4_silu{sfx}', f'gemm_moe2_fp4{sfx}', f'moe_sum_bf16_k{topk}']
            return Run(ks, ['{host:moe_fp4_host}', '{codir}', str(m), str(d), str(i), str(e), str(topk)], dict(BLOCK_M=bm, CHECK=2))
        out.append(Case(id=f'moe/fp4/M{m}D{d}I{i}E{e}K{topk}', family='moe', build=build, metric='us', parts=parts, ok=r'MOE_OK', timeout=900))
    return out


def conv():
    out = []
    for name, args in [('conv2d_bf16_r3s3_st1_p1_d1', '8 1 56 56 256 256 1 3 3 1 1 1'), ('conv2d_bf16_r1s1_st1_p0_d1', '8 1 56 56 256 512 1 1 1 1 0 1'),
                       ('conv3d_bf16_t3r3s3_st1_p1_d1', '2 16 28 28 128 256 3 3 3 1 1 1')]:
        def build(side, name=name, args=args):
            return Run([name], ['{host:conv_host}', '{co:%s}' % name, name] + args.split(), {})
        out.append(Case(id=f'conv/{name}', family='conv', build=build, metric='tflops', line=r'TF/s', ok=r'OK|PASS', timeout=600))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# GEMM: every operand kind with the commit's own tile / split-K rule (+ the aiter ASM vendor kernel where the host has one)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
GEMM_KINDS = [  # dispatch kind, host family of ours
    ('bf16', 'ours_v4_bf16'),
    ('i8_rowcol', 'ours_v4_i8'),
    ('fp8_rowcol', 'ours_v4_rowcol_fp8'),
    ('fp8_blockscale', 'ours_v4_fp8bs'),
]


def _tile(kernel):
    """(tileM, tileN) of a GEMM kernel by its name (the grid of vendor_gemm_host is (N / tileN, M / tileM))."""
    if kernel.endswith(('_64', '_64_splitk')):
        return 64, 128
    if kernel.endswith(('_128', '_128_splitk')) or kernel.startswith('gemm_a8w8_blockscale'):
        return 128, 128
    return 256, 256


def gemm():
    out = []
    shapes = [(4096, 4096, 8192), (8192, 8192, 8192), (1024, 7168, 2048), (256, 7168, 2048), (64, 8192, 8192), (16, 8192, 8192), (32, 4096, 14336)]
    # the shapes the rule changes of the composition search were fitted on (docs/compose-gemm-evidence.md) and hold-out shapes it was not fitted on (K = 6144, N = 8192, other M)
    shapes += [(2048, 7168, 7168), (1280, 7168, 4096), (192, 4096, 4096), (256, 4096, 8192), (128, 7168, 8192), (768, 4096, 8192)]
    shapes += [(320, 8192, 6144), (576, 7168, 6144), (1152, 4096, 6144), (1792, 8192, 4096), (160, 4096, 8192), (96, 8192, 8192)]
    for kind, fam in GEMM_KINDS:
        for (m, n, k) in shapes:
            def build(side, kind=kind, fam=fam, m=m, n=n, k=k):
                p = side.dispatch('gemm', kind, m, n, k)
                kn, s, red = p['kernel'], p['splits'], p['reduce']
                env = dict(SPLITK=s, ITERS=100, WARMUP=20, CHECK_ROWS=8)
                emit = [kn]
                if red:
                    env.update(REDUCE_CO='{co:%s}' % red, REDUCE_KN=red)
                    emit.append(red)
                tm, tn = _tile(kn)
                return Run(emit, ['{host:vendor_gemm_host}', fam, '{co:%s}' % kn, kn, str(m), str(n), str(k), str(tm), str(tn)], env)
            out.append(Case(id=f'gemm/{kind}/{m}x{n}x{k}', family='gemm', build=build, metric='us' if m <= 128 else 'tflops', line=r'ours|OURS|TF', ok=r'GEMM_OK', timeout=600))   # small M: the 1-decimal TF/s quantizes (see gemv)
    # the MXFP8 blockscale GEMM v4 against the aiter PR #6696 code object (gemm_ref_host runs both)
    for (m, n, k) in [(4096, 4096, 8192), (8192, 8192, 8192), (1024, 4096, 8192)]:
        def build(side, m=m, n=n, k=k):
            return Run(['gemm_mxfp8_bs_v4'], ['{host:gemm_ref_host}'], dict(GEMM_M=m, GEMM_N=n, GEMM_K=k, OURS_CO='{co:gemm_mxfp8_bs_v4}', OURS_KNAME='gemm_mxfp8_bs_v4', REF_CO=REF_CO.replace('ref_mxfp8', 'gemm_ref') if False else '',
                                                                          REF_ITERS=100, REF_WARMUP=100, REF_REPS=1, REF_ROWS=8, OURS_TILE_M=256, OURS_TILE_N=256))
        out.append(Case(id=f'gemm/mxfp8_v4/{m}x{n}x{k}', family='gemm', build=build, metric=r'timing OURS[^\n]*?median ([0-9.]+) TFLOP', better='high', line=r'timing OURS', ok='GEMM_OK', timeout=600))
    return out


def gemv():
    """Small-M GEMM / GEMV (W6): M in {1, 4, 16, 32, 64} x square N = K in {4096, 7168, 8192}, the dispatched kernel of each kind (weight-streaming
    bound: the report's TF/s converts to weight bytes per second as N K es / (2 M N K / TF/s))."""
    out = []
    for kind, fam in GEMM_KINDS:
        for nk in (4096, 7168, 8192):
            for m in (1, 4, 16, 32, 64):
                def build(side, kind=kind, fam=fam, m=m, n=nk, k=nk):
                    p = side.dispatch('gemm', kind, m, n, k)
                    kn, s, red = p['kernel'], p['splits'], p['reduce']
                    env = dict(SPLITK=s, ITERS=200, WARMUP=50, CHECK_ROWS=min(m, 8))
                    emit = [kn]
                    if red:
                        env.update(REDUCE_CO='{co:%s}' % red, REDUCE_KN=red)
                        emit.append(red)
                    tm, tn = _tile(kn)
                    return Run(emit, ['{host:vendor_gemm_host}', fam, '{co:%s}' % kn, kn, str(m), str(n), str(k), str(tm), str(tn)], env)
                # the host prints TF/s with one decimal (4.3 vs 4.2 TF/s is 2.4 % at M = 1): the small-M cases are judged on the microseconds it prints with two
                out.append(Case(id=f'gemv/{kind}/{m}x{nk}x{nk}', family='gemv', build=build, metric='us', line=r'ours|OURS|TF', ok=r'GEMM_OK', timeout=600))
    return out


SLOW_FOR_QUICK = r'gated_rmsnorm bf16 n128|moe_reduce k8 d7168|conv1d_fwd w4 silu B=128 L=512 uniform|argmax v32000|moe/fp8/'


def cases(tree, args):
    out = []
    for f in (membound, ):
        out += f(tree)
    for f in (attn_bf16, attn_hd192, attn_fp8pt, attn_mxfp8, attn_bwd, decode, mla, moe, conv, gemm, gemv):
        out += f()
    # the --quick set: per group (the first two id segments; a host for the memory-bound cases) the first case, and for the others also the last one
    groups = {}
    for c in out:
        key = c.id.split('{host:')[0] if False else ('/'.join(c.id.split('/')[:2]) if c.family != 'membound' else _mb_host(c))
        groups.setdefault(key, []).append(c)
    for key, cs in groups.items():
        cs[0].quick = True
        if cs[0].family != 'membound':
            cs[-1].quick = True
    for c in out:   # the hosts that initialise / check for minutes on the CPU stay in the full table only
        if re.search(SLOW_FOR_QUICK, c.id):
            c.quick = False
    return out


def _mb_host(c):
    r = c.build(None)
    return r.cmd[0]
