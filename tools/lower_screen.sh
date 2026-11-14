#!/usr/bin/env bash
# Static screen of a lowering-token experiment: every md5-manifest kernel emitted with and without AMDC_LOWER_ADD=<tokens>;
# per kernel: emit ok / fail, instruction count (VALU / SALU / total), md5 changed or not. Summary per family prefix.
#   bash tools/lower_screen.sh "<tokens or name:tokens;...>" [OUTDIR]   (AMDC=path to amdc; -P 16 parallel)
set -u
TOK="$1"; OUT="${2:-build/lscreen}"; A=${AMDC:-/tmp/target-kfmha/release/amdc}
mkdir -p "$OUT/base" "$OUT/exp"
cut -f1 tests/golden/md5_manifest.tsv | grep -v '^#' | grep -v '^$' > "$OUT/kernels.txt"
one() {
  k="$1"
  [ -s "$OUT/base/$k.s" ] || $A emit "$k" --out "$OUT/base/$k.s" > "$OUT/base/$k.log" 2>&1 || echo FAIL > "$OUT/base/$k.fail"
  AMDC_LOWER_ADD="$TOK" $A emit "$k" --out "$OUT/exp/$k.s" > "$OUT/exp/$k.log" 2>&1 || echo FAIL > "$OUT/exp/$k.fail"
}
export -f one; export OUT A TOK
xargs -P 16 -I{} bash -c 'one {}' < "$OUT/kernels.txt"
python3 - "$OUT" <<'EOF'
import os, re, sys, collections, hashlib
out = sys.argv[1]
ks = [l.strip() for l in open(f'{out}/kernels.txt') if l.strip()]
def count(p):
    if not os.path.exists(p): return None
    v = s = n = 0
    for l in open(p):
        t = l.strip()
        if not t or t.startswith(('.', ';', '/')) or t.endswith(':'): continue
        op = t.split()[0]
        if not re.match(r'^[sv]_|^ds_|^buffer_|^global_|^flat_', op): continue
        n += 1
        if op.startswith('v_'): v += 1
        elif op.startswith('s_') and not op.startswith(('s_nop', 's_waitcnt', 's_barrier')): s += 1
    return n, v, s
fam = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
fails = []
changed = 0
rows = []
for k in ks:
    b, e = count(f'{out}/base/{k}.s'), count(f'{out}/exp/{k}.s')
    if os.path.exists(f'{out}/exp/{k}.fail') and not os.path.exists(f'{out}/base/{k}.fail'):
        fails.append(k); continue
    if b is None or e is None: continue
    if open(f'{out}/base/{k}.s','rb').read() != open(f'{out}/exp/{k}.s','rb').read():
        changed += 1
    f = k.split('_')[0] + '_' + (k.split('_')[1] if '_' in k else '')
    a = fam[f]; a[0] += 1; a[1] += b[0]; a[2] += e[0]; a[3] += b[1]; a[4] += e[1]; a[5] += (e[0] != b[0])
    if e[0] != b[0]: rows.append((k, b, e))
print(f'kernels {len(ks)}, changed {changed}, NEW FAILURES {len(fails)}: {fails[:12]}')
for f, a in sorted(fam.items()):
    if a[5]: print(f'{f:28s} n={a[0]:3d} changed={a[5]:3d} insts {a[1]} -> {a[2]} ({a[2]-a[1]:+d})  valu {a[3]} -> {a[4]} ({a[4]-a[3]:+d})')
rows.sort(key=lambda r: r[2][0] - r[1][0])
for k, b, e in rows[:15]:
    print(f'  {k:52s} {b[0]} -> {e[0]} ({e[0]-b[0]:+d}) valu {b[1]} -> {e[1]} salu {b[2]} -> {e[2]}')
EOF
