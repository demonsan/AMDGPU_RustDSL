#!/usr/bin/env python3
"""Median-of-runs table for the batch-2 memory-bound matrix (see tools/membound_perf.sh).

usage: membound_perf_md.py run1.txt [run2.txt ...] --baseline docs/membound_baseline.tsv --out docs/membound-perf.md [--tol 0.10] [--stream 6500] [--update]
Each run file holds the lines of tools/b2_matrix.sh: `label | bad=0 OK | 1234.5 GB/s` (a missing GB/s = correctness-only case).
Exit status 1 when a case failed its correctness check or fell more than --tol below the baseline.
"""
import argparse, re, statistics, sys

LINE = re.compile(r'^(?P<label>.*?) \| (?P<bad>bad[a-z_]*=\d+) (?P<ok>OK|FAIL)(?: \| (?P<gbs>[0-9.]+) GB/s)?')

def parse(paths):
    runs = {}
    order = []
    for p in paths:
        for line in open(p, encoding='utf-8-sig', errors='replace'):
            m = LINE.match(line.strip())
            if not m:
                continue
            lab = m.group('label')
            if lab not in runs:
                runs[lab] = {'gbs': [], 'ok': True}
                order.append(lab)
            if m.group('ok') != 'OK' or not m.group('bad').endswith('=0'):
                runs[lab]['ok'] = False
            if m.group('gbs'):
                runs[lab]['gbs'].append(float(m.group('gbs')))
    return order, runs

def read_baseline(path):
    base = {}
    try:
        for line in open(path, encoding='utf-8-sig'):
            if line.startswith('#') or '\t' not in line:
                continue
            lab, v = line.rstrip('\n').split('\t')[:2]
            base[lab] = float(v)
    except FileNotFoundError:
        pass
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tol', type=float, default=0.10)
    ap.add_argument('--stream', type=float, default=6500.0)
    ap.add_argument('--update', action='store_true')
    a = ap.parse_args()
    order, runs = parse(a.runs)
    base = read_baseline(a.baseline)
    rows, bad, regress = [], 0, 0
    for lab in order:
        r = runs[lab]
        med = statistics.median(r['gbs']) if r['gbs'] else None
        b = base.get(lab)
        status = 'ok'
        if not r['ok']:
            status = 'CORRECTNESS FAIL'; bad += 1
        elif med is not None and b:
            d = med / b - 1.0
            if d < -a.tol:
                status = 'REGRESSION'; regress += 1
            elif d > a.tol:
                status = 'faster'
        elif med is not None and not b:
            status = 'new'
        rows.append((lab, med, b, status))
    lines = ['# Memory-bound perf matrix', '',
             f'Median of {len(a.runs)} run(s) of `tools/b2_matrix.sh` (one GPU, same node, same session); the baseline is `docs/membound_baseline.tsv`; "% stream" is against {a.stream:.0f} GB/s (the measured `nt` stream copy). '
             'Cases that count bytes read from cache several times (the MoE bias rows) or are latency-bound small problems can sit far from the stream line by construction; the point of the table is the delta column: '
             f'a drop of more than {a.tol * 100:.0f} % against the baseline is flagged (node noise is +-3-10 %).', '',
             '| case | GB/s | % stream | baseline | delta | status |', '|---|---:|---:|---:|---:|---|']
    for lab, med, b, st in rows:
        if med is None:
            lines.append(f'| {lab} | - | - | - | - | {st} |')
        else:
            pct = f'{100 * med / a.stream:.0f} %'
            lines.append(f'| {lab} | {med:.0f} | {pct} | {b:.0f} | {100 * (med / b - 1):+.1f} % | {st} |' if b else f'| {lab} | {med:.0f} | {pct} | - | - | {st} |')
    lines += ['', f'{len(rows)} cases, {bad} correctness failure(s), {regress} regression(s).']
    open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
    if a.update:
        with open(a.baseline, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# label<TAB>GB/s (median of the run that wrote it; regenerate with: bash tools/membound_perf.sh update)\n')
            for lab, med, b, st in rows:
                if med is not None:
                    f.write(f'{lab}\t{med:.1f}\n')
    print('\n'.join(l for l in lines if ('REGRESSION' in l or 'FAIL' in l or l.startswith(str(len(rows))))))
    return 1 if (bad or regress) else 0

if __name__ == '__main__':
    sys.exit(main())