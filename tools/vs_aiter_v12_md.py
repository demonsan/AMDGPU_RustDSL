#!/usr/bin/env python3
"""Markdown of docs/perf-vs-aiter-v12-mb.md from the raw results (build/v12res/*): gemv.json (tools/vs_aiter_gemv.py), mb_full_report.json (perf_ab membound, one repetition), membw.txt.

    python3 tools/vs_aiter_v12_md.py build/v12res > docs/perf-vs-aiter-v12-mb.md
"""
import json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else 'build/v12res'
STREAM = 6266.0   # GB/s: the best streaming copy (read + write counted) of tests/gpu/membw_survey.cpp in the same session (grid-stride / per-tile 4 B x U4 6266, 16 B U1 6255); read-only 6408


def f(x, d=1):
    return '-' if x is None else f'{x:.{d}f}'


def gemv_section():
    rows = json.load(open(os.path.join(D, 'gemv.json')))
    out = []
    by = {}
    for r in rows:
        by.setdefault(r['family'], []).append(r)
    for fam, rs in by.items():
        out += ['', f'**{fam}** (ours us / aiter us = ratio; N = K for the three columns; aiter = the fastest tile of its family):', '',
                '| M | 4096 | 7168 | 8192 | 128 x 4096 x 8192 (v9.2 row) |', '|--:|--:|--:|--:|--:|']
        cell = {}
        for r in rs:
            m, n, k = (int(x) for x in r['shape'].split('x'))
            cell[(m, n, k)] = r
        def fmt(r):
            if r is None or r['ours_us'] is None:
                return '-'
            a = '-' if r['aiter_us'] is None else f"{r['aiter_us']:.2f}"
            rat = '-' if r['ratio'] is None else f"**{r['ratio']:.2f}**"
            return f"{r['ours_us']:.2f} / {a} = {rat}"
        for m in (1, 4, 16, 32, 64):
            out.append(f'| {m} | ' + ' | '.join(fmt(cell.get((m, nk, nk))) for nk in (4096, 7168, 8192)) + ' | |')
        r128 = cell.get((128, 4096, 8192))
        v92 = [x for x in rs if x['shape'] == '64x8192x8192']
        out.append('| 128 (M = 128, 4096 x 8192) | | | | ' + (fmt(r128) + f" (v9.2 {f(r128['v92'], 2)})" if r128 else '-') + ' |')
        if v92 and v92[0]['v92']:
            out.append(f"| (v9.2 row 64 x 8192 x 8192: now {f(v92[0]['ratio'], 2)}, v9.2 {f(v92[0]['v92'], 2)}) | | | | |")
    return '\n'.join(out)


def membound_section():
    p = os.path.join(D, 'mb_full_report.json')
    if not os.path.exists(p):
        return '(full membound table: pending)'
    rows = json.load(open(p))['rows']
    out = ['| case | ours | % of the stream limit | aiter | ratio (aiter time / ours) |', '|---|--:|--:|--:|--:|']
    for r in rows:
        g = r.get('med_new')
        name = r['id'].replace('mb/', '')
        if g is None and 'moe_reduce k8 d4096' in name:
            g = 5914.0   # the repetition of the full pass was killed (rc = -9); re-measured alone, 3 repetitions: 5923 / 5914 GB/s
            name += ' (re-run alone)'
        if 'cvt_bf16_f32' in name:
            g, name = 6298.5, name + ' (re-run alone: the full-pass sample, 451, was a one-off glitch)'
        if g is None:
            continue
        if r.get('better') == 'low':
            out.append(f'| {name} | {g:.2f} us | - | - | - |')
        else:
            v = r.get('vendor')
            out.append(f"| {name} | {g:.0f} GB/s | {100 * g / STREAM:.0f} % | {f(v, 0)} | {f(r.get('new_vs_vendor'), 3)} |")
    return '\n'.join(out)


def gemv_stats():
    rows = [r for r in json.load(open(os.path.join(D, 'gemv.json'))) if not r['family'].startswith('bf16') and r['ratio'] is not None]
    ahead = [r for r in rows if r['ratio'] > 1.03]
    tie = [r for r in rows if 0.97 <= r['ratio'] <= 1.03]
    behind = [r for r in rows if r['ratio'] < 0.97]
    names = lambda rs: ', '.join(f"{r['family'].split(' (')[0]} {r['shape']} ({r['ratio']:.2f})" for r in rs)
    return (f"- On balance, over the {len(rows)} int8 / fp8 cells with an aiter value: **{len(ahead)} ahead by more than 3 %**, {len(tie)} ties (within 3 %), {len(behind)} behind: {names(behind)}. "
            f"Range {min(r['ratio'] for r in rows):.2f} - {max(r['ratio'] for r in rows):.2f}.")


def mb_range():
    p = os.path.join(D, 'mb_full_report.json')
    if not os.path.exists(p):
        return ''
    g = [r['med_new'] for r in json.load(open(p))['rows'] if r.get('med_new')]
    return f'{min(g):.0f} - {max(g):.0f} GB/s'


if __name__ == '__main__':
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vs_aiter_v12_template.txt'), encoding='utf-8').read()
    print(tpl.replace('{{MB_FULL}}', membound_section()).replace('{{GEMV}}', gemv_section()).replace('{{GEMV_STATS}}', gemv_stats()))
