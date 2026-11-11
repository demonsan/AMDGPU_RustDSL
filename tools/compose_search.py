#!/usr/bin/env python3
"""Measured composition search for dense GEMM (docs/compose.md): enumerate -> prune -> compile-only feasibility -> MEASURED top-k, then compare with the recorded dispatch rule.

    python3 tools/compose_search.py search gemm <kind> <M> <N> <K> [--top 3] [--reps 5] [--margin 1.5]
    python3 tools/compose_search.py check-gemm [--kinds a,b] [--reps 5] [--tol 0.03] [--top 3] [--out build/compose/gemm.md]

`search`: `amdc compose-search gemm ...` lists the feasible candidates (tile variant x split-K, with the reduce kernel) and the coarse model's pruning; the `--top` best surviving candidates by model plus the
recorded rule's own choice are emitted, checked with `amdc summary` (VGPR / AGPR / LDS: a kernel that does not compile or exceeds the budgets is infeasible) and measured INTERLEAVED (per repetition every
candidate once, order rotating) on `vendor_gemm_host`; the median per candidate decides. `check-gemm` runs that over the shapes the dispatch rules record their evidence on (`dispatch.rs` doc
comments) and prints where the recorded rule is NOT within `--tol` of the measured best: a rule whose evidence a newer compiler no longer supports shows up here. GPUs 0-3 only (tools/gpu_guard.sh rule).
Environment: AMDC (default target/debug/amdc), ROCM_PATH, CLANG, HIP_VISIBLE_DEVICES.
"""
import argparse, json, os, re, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_ab as pa  # noqa: E402  (gpu_guard, tool paths, host build)

ROOT = pa.ROOT
FAM = {'bf16': 'ours_v4_bf16', 'i8_rowcol': 'ours_v4_i8', 'fp8_rowcol': 'ours_v4_rowcol_fp8', 'fp8_blockscale': 'ours_v4_fp8bs'}
# shapes the rules in dispatch.rs record their same-run evidence on (M, N, K)
EVIDENCE = [(4096, 4096, 4096), (2048, 4096, 4096), (1024, 4096, 8192), (512, 7168, 4096), (256, 7168, 4096), (128, 4096, 8192), (64, 8192, 8192), (512, 7168, 7168), (2048, 7168, 7168)]


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
        self.host = None
        self.emitted = {}
        self.summary = {}

    def build_host(self):
        if self.host is None:
            name, exe, msg = pa.build_host(ROOT, 'vendor_gemm_host', self.out)
            if not exe:
                raise RuntimeError('vendor_gemm_host build failed: ' + msg)
            self.host = exe
        return self.host

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
    """(tileM, tileN) of the kernel (the grid of vendor_gemm_host is (N / tileN, M / tileM)): `_64*` 64 x 128, `_128*` / `_splitk` of the block-scale kind / its base 128 x 128, else 256 x 256."""
    if kernel.endswith(('_64', '_64_splitk')):
        return 64, 128
    if kernel.endswith(('_128', '_128_splitk')) or kernel.startswith('gemm_a8w8_blockscale'):
        return 128, 128
    return 256, 256


def measure_once(work, kind, m, n, k, cand):
    co, _ = work.kernel(cand['kernel'])
    env = dict(os.environ, SPLITK=str(cand['splits']), ITERS='100', WARMUP='20', CHECK_ROWS='8')
    if cand.get('reduce'):
        rco, _ = work.kernel(cand['reduce'])
        env.update(REDUCE_CO=rco, REDUCE_KN=cand['reduce'])
    tm, tn = tile_args(cand['kernel'])
    p = subprocess.run([work.build_host(), FAM[kind], co, cand['kernel'], str(m), str(n), str(k), str(tm), str(tn)], capture_output=True, text=True, env=env, timeout=600, errors='replace')
    text = p.stdout + p.stderr
    mt = re.search(r'([0-9.]+) us\s+([0-9.]+) TF/s', text)
    ok = 'GEMM_OK' in text
    return (float(mt.group(1)) if mt else None), ok


def search(work, kind, m, n, k, top, reps, margin):
    js = json.loads(run_amdc('compose-search', 'gemm', kind, m, n, k, '--margin', margin))
    cands = js['candidates']
    rule = js['rule']
    keep = sorted([c for c in cands if c['kept']], key=lambda c: c['model_us'])[:top]
    # the recorded rule's own choice is always measured (it may not survive the model)
    def same(c, r):
        return r and c['kernel'] == r['kernel'] and c['splits'] == r['splits']
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
        rows.append(dict(c, feasible=okf, why=why, samples=[], ok=True, is_rule=bool(same(c, rule))))
    live = [r for r in rows if r['feasible']]
    for rep in range(reps):
        order = live[rep % len(live):] + live[:rep % len(live)] if live else []
        for r in order:
            us, ok = measure_once(work, kind, m, n, k, r)
            if us is not None:
                r['samples'].append(us)
            r['ok'] = r['ok'] and ok
    for r in rows:
        r['median_us'] = statistics.median(r['samples']) if r['samples'] else None
    return dict(kind=kind, m=m, n=n, k=k, rule=rule, rows=rows, pruned=len([c for c in cands if not c['kept']]), total=len(cands))


def best_and_rule(res):
    ok = [r for r in res['rows'] if r['median_us'] and r['ok']]
    best = min(ok, key=lambda r: r['median_us']) if ok else None
    rule = next((r for r in res['rows'] if r['is_rule']), None)
    return best, rule


def fmt_rows(res):
    best, rule = best_and_rule(res)
    out = [f"### {res['kind']} {res['m']}x{res['n']}x{res['k']}: {res['total']} candidates, {res['pruned']} pruned by the model; rule = `{res['rule']['kernel']}` S{res['rule']['splits']}", '',
           '| candidate | splits | tile | ctas | model us | measured us | feasible | |', '|---|--:|---|--:|--:|--:|---|---|']
    for r in sorted(res['rows'], key=lambda r: (r['median_us'] is None, r['median_us'] or 0)):
        tag = ' '.join(t for t, c in (('**best**', r is best), ('rule', r['is_rule'])) if c)
        mu = '-' if r['median_us'] is None else f"{r['median_us']:.2f}"
        out.append(f"| `{r['kernel']}` | {r['splits']} | {r['tile'][0]}x{r['tile'][1]} | {r['ctas']} | {r['model_us']:.1f} | {mu} | {'yes' if r['feasible'] else 'NO: ' + str(r['why'])} | {tag} |")
    return out, best, rule


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('family', choices=['gemm'])
    s.add_argument('kind', choices=list(FAM))
    s.add_argument('m', type=int)
    s.add_argument('n', type=int)
    s.add_argument('k', type=int)
    c = sub.add_parser('check-gemm')
    c.add_argument('--kinds', default=','.join(FAM))
    for p in (s, c):
        p.add_argument('--top', type=int, default=3)
        p.add_argument('--reps', type=int, default=5)
        p.add_argument('--margin', type=float, default=1.5)
        p.add_argument('--out', default=os.path.join(ROOT, 'build', 'compose'))
    c.add_argument('--tol', type=float, default=0.03, help='the rule counts as confirmed when its measured time is within this fraction of the measured best')
    a = ap.parse_args()
    pa.gpu_guard()
    work = Work(a.out)
    if a.cmd == 'search':
        res = search(work, a.kind, a.m, a.n, a.k, a.top, a.reps, a.margin)
        lines, best, rule = fmt_rows(res)
        print('\n'.join(lines))
        return 0
    report = ['# composition search: GEMM (measured top-k vs the recorded dispatch rule)', '', f'{a.reps} interleaved repetitions, top {a.top} by the coarse model + the rule\'s choice; tolerance {a.tol:.0%}.', '']
    disagree = []
    summary = []
    for kind in a.kinds.split(','):
        for (m, n, k) in EVIDENCE:
            try:
                res = search(work, kind, m, n, k, a.top, a.reps, a.margin)
            except Exception as e:  # shape not valid for the kind (e.g. no candidate): record and go on
                summary.append(f'| {kind} | {m}x{n}x{k} | - | - | - | skipped: {str(e)[:60]} |')
                continue
            lines, best, rule = fmt_rows(res)
            report += lines + ['']
            if not best or not rule or rule['median_us'] is None:
                summary.append(f'| {kind} | {m}x{n}x{k} | - | - | - | no measurement |')
                continue
            gap = rule['median_us'] / best['median_us'] - 1
            verdict = 'confirmed' if gap <= a.tol else 'DISAGREES'
            if gap > a.tol:
                disagree.append((kind, m, n, k, rule['kernel'], rule['splits'], best['kernel'], best['splits'], gap))
            summary.append(f"| {kind} | {m}x{n}x{k} | `{rule['kernel']}` S{rule['splits']} {rule['median_us']:.1f} us | `{best['kernel']}` S{best['splits']} {best['median_us']:.1f} us | {gap:+.1%} | {verdict} |")
    head = ['## Summary', '', '| kind | shape | recorded rule (measured) | measured best | rule vs best | verdict |', '|---|---|---|---|--:|---|'] + summary + ['', f'**{len(disagree)} shape(s) where the recorded rule is not within {a.tol:.0%} of the measured best.**', '']
    text = '\n'.join(report[:4] + head + report[4:]) + '\n'
    path = os.path.join(a.out, 'gemm.md')
    open(path, 'w').write(text)
    print(text)
    return 1 if disagree else 0


if __name__ == '__main__':
    sys.exit(main())
