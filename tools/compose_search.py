#!/usr/bin/env python3
"""Measured composition search (docs/compose.md): enumerate -> prune -> compile-only feasibility -> MEASURED top-k, then compare with the recorded dispatch rule.

    python3 tools/compose_search.py search gemm <kind> <M> <N> <K> [--top 3] [--reps 5] [--margin 1.5] [--stat median|min]
    python3 tools/compose_search.py search decode16 <B> <HK> <CTX> [--page 16]
    python3 tools/compose_search.py search prefill_bf16 <B> <H> <SQ> <SK> <causal 0|1>
    python3 tools/compose_search.py check-gemm [--kinds a,b] [--ms .. --ns .. --ks ..] [--reps 5] [--tol 0.03] [--top 3] [--out build/compose] [--name gemm]
    python3 tools/compose_search.py check-decode16 [--shapes "b,hk,ctx;..."]      python3 tools/compose_search.py check-prefill [--shapes "b,h,sq,sk,causal;..."]

`search`: `amdc compose-search <family> ...` lists the feasible candidates (GEMM: tile variant x split-K with the reduce kernel; decode16: fused or last-arriver with S key splits; prefill_bf16: plain /
paired / q128 / `_ks` with the combine kernel) and the coarse model's pruning; the `--top` best surviving candidates by model plus the recorded rule's own choice are emitted, checked with `amdc summary`
(VGPR / AGPR / LDS: a kernel that does not compile or exceeds the budgets is infeasible) and measured INTERLEAVED (per repetition every candidate once, order rotating) on the family's host
(`vendor_gemm_host`, `pa_host`, `attn_host`); the estimator per candidate is the median of the repetitions (`--stat min` = the best repetition: interference on a shared node only ever ADDS time, so the
minimum is the interference-free estimate; use it only when the node cannot be made quiet, and say so with the result). `check-*` re-run the search over the shapes the rules record their evidence on and
list where the recorded rule is NOT within `--tol` of the measured best. GPUs 0-3 only (tools/gpu_guard.sh rule). Environment: AMDC (default target/debug/amdc), ROCM_PATH, CLANG, HIP_VISIBLE_DEVICES.
"""
import argparse, json, math, os, re, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_ab as pa  # noqa: E402  (gpu_guard, tool paths, host build)

ROOT = pa.ROOT
FAM = {'bf16': 'ours_v4_bf16', 'i8_rowcol': 'ours_v4_i8', 'fp8_rowcol': 'ours_v4_rowcol_fp8', 'fp8_blockscale': 'ours_v4_fp8bs'}
# shapes the rules in dispatch.rs record their same-run evidence on
EVIDENCE = [(4096, 4096, 4096), (2048, 4096, 4096), (1024, 4096, 8192), (512, 7168, 4096), (256, 7168, 4096), (128, 4096, 8192), (64, 8192, 8192), (512, 7168, 7168), (2048, 7168, 7168)]
DECODE_EVIDENCE = [(b, 8, ctx) for b in (4, 8, 16, 32, 64) for ctx in (1024, 2048, 8192)]
PREFILL_EVIDENCE = [(1, 8, 4096, 4096, 0), (1, 8, 4096, 4096, 1), (1, 8, 2048, 2048, 0), (1, 8, 2048, 2048, 1), (2, 8, 1000, 1300, 0), (2, 8, 1000, 1300, 1), (1, 32, 8192, 8192, 1), (4, 16, 2048, 2048, 0)]
STAT = 'median'


def amdc():
    return os.environ.get('AMDC') or os.path.join(ROOT, 'target', 'debug', 'amdc')


def run_amdc(*args):
    p = pa.sh([amdc(), *map(str, args)], cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError(f'amdc {" ".join(map(str, args))}: {(p.stderr or p.stdout).strip()[:300]}')
    return p.stdout


class Work:
    def __init__(self, out):
        self.out = os.path.abspath(out)
        os.makedirs(os.path.join(self.out, 'co'), exist_ok=True)
        self.hosts = {}
        self.emitted = {}
        self.summary = {}

    def host(self, name):
        if name not in self.hosts:
            n, exe, msg = pa.build_host(ROOT, name, self.out)
            if not exe:
                raise RuntimeError(f'{name} build failed: ' + msg)
            self.hosts[name] = exe
        return self.hosts[name]

    def kernel(self, k):
        """Emit + assemble once; returns (co path, summary dict or error string)."""
        if k in self.emitted:
            return self.emitted[k], self.summary[k]
        co = os.path.join(self.out, 'co', k + '.co')
        s = os.path.join(self.out, 'co', k + '.s')
        p = pa.sh([amdc(), 'emit', k, '--backend', 'own', '--out', s], cwd=ROOT, timeout=1200)
        if p.returncode != 0 or not os.path.exists(s):
            self.emitted[k], self.summary[k] = None, 'emit failed: ' + (p.stdout + p.stderr)[-200:]
            return None, self.summary[k]
        a = pa.sh([pa.CLANG, '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mcode-object-version=6', '-o', co, s], timeout=600)
        if a.returncode != 0:
            self.emitted[k], self.summary[k] = None, 'assemble failed: ' + (a.stdout + a.stderr)[-200:]
            return None, self.summary[k]
        sm = pa.sh([amdc(), 'summary', k, '--json'], cwd=ROOT, timeout=600)
        try:
            self.summary[k] = json.loads(sm.stdout)
        except Exception:
            self.summary[k] = {}
        self.emitted[k] = co
        return co, self.summary[k]


def feasible(sm):
    """Compile-only feasibility from `amdc summary`: the kernel compiled (summary present) and its registers fit (arch VGPR + AGPR <= 512 per lane, LDS <= 160 KiB)."""
    if isinstance(sm, str):
        return False, sm
    if not sm:
        return True, 'no summary'
    v, a, lds = sm.get('vgpr', 0), sm.get('agpr', 0), sm.get('lds_bytes', 0)
    if v + a > 512 or lds > 160 * 1024:
        return False, f'vgpr {v} agpr {a} lds {lds}'
    return True, f'vgpr {v} agpr {a} lds {lds}'


def tile_args(kernel):
    """(tileM, tileN) of a GEMM kernel (the grid of vendor_gemm_host is (N / tileN, M / tileM)): `_64*` 64 x 128, `_128*` / the block-scale kinds 128 x 128, else 256 x 256."""
    if kernel.endswith(('_64', '_64_splitk')):
        return 64, 128
    if kernel.endswith(('_128', '_128_splitk')) or kernel.startswith('gemm_a8w8_blockscale'):
        return 128, 128
    return 256, 256


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# families: the amdc arguments, the host command of one candidate, the metric
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def fam_amdc_args(fam, shape):
    if fam == 'gemm':
        kind, m, n, k = shape
        return ['gemm', kind, m, n, k]
    if fam == 'decode16':
        b, hk, ctx, page = shape
        return ['decode16', b, hk, ctx, '--page', page]
    b, h, sq, sk, c = shape
    return ['prefill_bf16', b, h, sq, sk, c]


def shape_label(fam, shape):
    return ' '.join(str(x) for x in shape) if fam != 'gemm' else f'{shape[0]} {shape[1]}x{shape[2]}x{shape[3]}'


def measure_once(work, fam, shape, cand):
    """One run of one candidate: (microseconds, correct)."""
    if fam == 'gemm':
        kind, m, n, k = shape
        co, _ = work.kernel(cand['kernel'])
        env = dict(os.environ, SPLITK=str(cand['splits']), ITERS='100', WARMUP='20', CHECK_ROWS='8')
        if cand.get('reduce'):
            rco, _ = work.kernel(cand['reduce'])
            env.update(REDUCE_CO=rco, REDUCE_KN=cand['reduce'])
        tm, tn = tile_args(cand['kernel'])
        cmd = [work.host('vendor_gemm_host'), FAM[kind], co, cand['kernel'], str(m), str(n), str(k), str(tm), str(tn)]
        ok_re, us_re = 'GEMM_OK', r'([0-9.]+) us\s+([0-9.]+) TF/s'
    elif fam == 'decode16':
        b, hk, ctx, page = shape
        co, _ = work.kernel(cand['kernel'])
        cco, _ = work.kernel('attn_combine_bf16_hd128')
        env = dict(os.environ, KPREFIX='pa_decode16_bf16_hd128_p')
        env.update(dict(FUSED='1', KSUFFIX='_f') if cand['kernel'].endswith('_f') else dict(LASTC='1', KSUFFIX='_c'))
        cmd = [work.host('pa_host'), 'ours', co, cco, str(b), str(hk * 8), str(hk), str(ctx), str(page), str(cand['splits'])]
        ok_re, us_re = 'PA_OK', r'^ours .*?([0-9.]+) us'
    else:
        b, h, sq, sk, c = shape
        k = cand['kernel']
        co, _ = work.kernel(k)
        rows = 128 if (k.endswith('_q128') or k == 'fmha_fwd_hd128_bf16_causal') else 256
        pair = 0 if k == 'fmha_fwd_hd128_bf16_causal' else int(c)
        env = dict(os.environ, LSE='0', CHECK='4', ROWS=str(rows), BLOCK=str(rows * 2), PAIR=str(pair))
        if cand['splits'] > 1:
            cco, _ = work.kernel('attn_combine_bf16_hd128_q16')
            env.update(KS=str(int(round(math.log2(cand['splits'])))), COMBINE_CO=cco)
        cmd = [work.host('attn_host'), 'ours', co, k, str(b), str(h), str(h), str(sq), str(sk), str(int(c))]
        ok_re, us_re = 'ATTN_OK', r'^ours .*?([0-9.]+)\s*TF/s'
        flops = 4.0 * b * h * sq * sk * 128 * (0.5 if c else 1.0)   # the host reports TF/s: convert to microseconds for one launch
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, errors='replace')
    text = p.stdout + p.stderr
    mt = re.search(us_re, text, re.M)
    if mt and fam == 'prefill_bf16':
        return flops / (float(mt.group(1)) * 1e6), (ok_re in text)
    return (float(mt.group(1)) if mt else None), (ok_re in text)


def search(work, fam, shape, top, reps, margin):
    js = json.loads(run_amdc('compose-search', *fam_amdc_args(fam, shape), '--margin', margin))
    cands = js['candidates']
    rule = js['rule']
    keep = sorted([c for c in cands if c['kept']], key=lambda c: c['model_us'])[:top]

    def same(c, r):
        return bool(r) and c['kernel'] == r['kernel'] and c['splits'] == r['splits']
    # the recorded rule's own choice is always measured (it may not survive the model)
    if rule and not any(same(c, rule) for c in keep):
        keep += [c for c in cands if same(c, rule)]
    rows = []
    for c in keep:
        co, sm = work.kernel(c['kernel'])
        okf, why = feasible(sm) if co else (False, sm)
        if c.get('reduce') and okf:
            rco, rsm = work.kernel(c['reduce'])
            if not rco:
                okf, why = False, 'reduce kernel: ' + str(rsm)
        rows.append(dict(c, feasible=okf, why=why, samples=[], ok=True, is_rule=same(c, rule)))
    live = [r for r in rows if r['feasible']]
    for rep in range(reps):
        order = live[rep % len(live):] + live[:rep % len(live)] if live else []
        for r in order:
            us, ok = measure_once(work, fam, shape, r)
            if us is not None:
                r['samples'].append(us)
            r['ok'] = r['ok'] and ok
    for r in rows:
        r['median_us'] = (min(r['samples']) if STAT == 'min' else statistics.median(r['samples'])) if r['samples'] else None
    return dict(fam=fam, shape=shape, rule=rule, rows=rows, pruned=len([c for c in cands if not c['kept']]), total=len(cands))


def best_and_rule(res):
    ok = [r for r in res['rows'] if r['median_us'] and r['ok']]
    best = min(ok, key=lambda r: r['median_us']) if ok else None
    rule = next((r for r in res['rows'] if r['is_rule']), None)
    return best, rule


def fmt_rows(res):
    best, rule = best_and_rule(res)
    out = [f"### {res['fam']} {shape_label(res['fam'], res['shape'])}: {res['total']} candidates, {res['pruned']} pruned by the model; rule = `{res['rule']['kernel']}` S{res['rule']['splits']}", '',
           '| candidate | splits | tile | ctas | model us | measured us | feasible | |', '|---|--:|---|--:|--:|--:|---|---|']
    for r in sorted(res['rows'], key=lambda r: (r['median_us'] is None, r['median_us'] or 0)):
        tag = ' '.join(t for t, c in (('**best**', r is best), ('rule', r['is_rule'])) if c)
        mu = '-' if r['median_us'] is None else f"{r['median_us']:.2f} ({min(r['samples']):.1f}-{max(r['samples']):.1f})"
        tile = f"{r['tile'][0]}x{r['tile'][1]}" if r['tile'][0] else '-'
        out.append(f"| `{r['kernel']}` | {r['splits']} | {tile} | {r['ctas']} | {r['model_us']:.1f} | {mu} | {'yes' if r['feasible'] else 'NO: ' + str(r['why'])} | {tag} |")
    return out, best, rule


def check(work, fam, shapes, a, title):
    report = [f'# composition search: {title} (measured top-k vs the recorded dispatch rule)', '',
              f'{a.reps} interleaved repetitions, top {a.top} by the coarse model + the rule\'s choice; estimator: {"best repetition (--stat min)" if STAT == "min" else "median"}; tolerance {a.tol:.0%}.', '']
    summary, results, disagree = [], [], []
    for shape in shapes:
        try:
            res = search(work, fam, shape, a.top, a.reps, a.margin)
        except Exception as e:  # shape not valid for the family (no candidate): record and go on
            summary.append(f'| {shape_label(fam, shape)} | - | - | - | skipped: {str(e)[:60]} |')
            continue
        lines, best, rule = fmt_rows(res)
        report += lines + ['']
        results.append(dict(fam=fam, shape=shape, rule=res['rule'], rows=[{x: r[x] for x in ('kernel', 'splits', 'tile', 'ctas', 'median_us', 'is_rule', 'feasible', 'samples')} for r in res['rows']]))
        if not best or not rule or rule['median_us'] is None:
            summary.append(f'| {shape_label(fam, shape)} | - | - | - | no measurement |')
            continue
        gap = rule['median_us'] / best['median_us'] - 1
        if gap > a.tol:
            disagree.append((shape, rule['kernel'], rule['splits'], best['kernel'], best['splits'], gap))
        summary.append(f"| {shape_label(fam, shape)} | `{rule['kernel']}` S{rule['splits']} {rule['median_us']:.1f} us | `{best['kernel']}` S{best['splits']} {best['median_us']:.1f} us | {gap:+.1%} | {'confirmed' if gap <= a.tol else 'DISAGREES'} |")
    head = ['## Summary', '', '| shape | recorded rule (measured) | measured best | rule vs best | verdict |', '|---|---|---|--:|---|'] + summary + ['', f'**{len(disagree)} shape(s) where the recorded rule is not within {a.tol:.0%} of the measured best.**', '']
    return report[:4] + head + report[4:], results, disagree


def main():
    global STAT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('family', choices=['gemm', 'decode16', 'prefill_bf16'])
    s.add_argument('args', nargs='+')
    s.add_argument('--page', type=int, default=16)
    cg = sub.add_parser('check-gemm')
    cg.add_argument('--kinds', default=','.join(FAM))
    cg.add_argument('--ms', default='')
    cg.add_argument('--ns', default='')
    cg.add_argument('--ks', default='')
    cd = sub.add_parser('check-decode16')
    cd.add_argument('--shapes', default='', help='"b,hk,ctx;..." (default: the evidence grid of dispatch::decode16_plan)')
    cd.add_argument('--page', type=int, default=16)
    cpf = sub.add_parser('check-prefill')
    cpf.add_argument('--shapes', default='', help='"b,h,sq,sk,causal;..." (default: the evidence shapes of dispatch::bf16_prefill_kernel)')
    for p in (s, cg, cd, cpf):
        p.add_argument('--top', type=int, default=3)
        p.add_argument('--reps', type=int, default=5)
        p.add_argument('--margin', type=float, default=1.5)
        p.add_argument('--stat', choices=['median', 'min'], default='median', help='the per-candidate estimator over the repetitions (default median; min on a shared / noisy node: say so with the result)')
        p.add_argument('--out', default=os.path.join(ROOT, 'build', 'compose'))
    for p in (cg, cd, cpf):
        p.add_argument('--tol', type=float, default=0.03, help='the rule counts as confirmed when its measured time is within this fraction of the measured best')
        p.add_argument('--name', default='')
    a = ap.parse_args()
    pa.gpu_guard()
    STAT = a.stat
    work = Work(a.out)
    if a.cmd == 'search':
        if a.family == 'gemm':
            shape = (a.args[0],) + tuple(int(x) for x in a.args[1:4])
        elif a.family == 'decode16':
            shape = tuple(int(x) for x in a.args[:3]) + (a.page,)
        else:
            shape = tuple(int(x) for x in a.args[:5])
        res = search(work, a.family, shape, a.top, a.reps, a.margin)
        print('\n'.join(fmt_rows(res)[0]))
        return 0
    if a.cmd == 'check-gemm':
        shapes = EVIDENCE
        if a.ms and a.ns and a.ks:
            shapes = [(int(m), int(n), int(k)) for m in a.ms.split(',') for n in a.ns.split(',') for k in a.ks.split(',')]
        lines, results, dis = [], [], []
        for kind in a.kinds.split(','):
            l, r, d = check(work, 'gemm', [(kind,) + sh for sh in shapes], a, f'GEMM {kind}')
            lines += l + ['']
            results += r
            dis += d
        name = a.name or 'gemm'
    elif a.cmd == 'check-decode16':
        shapes = [tuple(int(x) for x in t.split(',')) + (a.page,) for t in a.shapes.split(';')] if a.shapes else [sh + (a.page,) for sh in DECODE_EVIDENCE]
        lines, results, dis = check(work, 'decode16', shapes, a, 'paged decode (decode16)')
        name = a.name or 'decode16'
    else:
        shapes = [tuple(int(x) for x in t.split(',')) for t in a.shapes.split(';')] if a.shapes else PREFILL_EVIDENCE
        lines, results, dis = check(work, 'prefill_bf16', shapes, a, 'bf16 hd128 prefill')
        name = a.name or 'prefill_bf16'
    text = '\n'.join(lines) + '\n'
    open(os.path.join(a.out, name + '.md'), 'w').write(text)
    json.dump(results, open(os.path.join(a.out, name + '.json'), 'w'), indent=1)
    print(text)
    return 1 if dis else 0


if __name__ == '__main__':
    sys.exit(main())
