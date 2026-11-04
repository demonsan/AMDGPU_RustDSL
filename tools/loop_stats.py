import re, sys
# per loop (back edge to an earlier label): instruction count, constant v_movs, sgpr->vgpr v_movs inside the loop range
lines = open(sys.argv[1]).read().split("\n")
labels = {}
for i, l in enumerate(lines):
    m = re.match(r"^(BB\d+):", l)
    if m:
        labels[m.group(1)] = i
out = []
for i, l in enumerate(lines):
    m = re.match(r"^\s+s_(?:cbranch_\w+|branch)\s+(BB\d+)", l)
    if m and labels.get(m.group(1), 10**9) < i:
        a = labels[m.group(1)]
        body = lines[a:i + 1]
        ins = [x for x in body if re.match(r"^\s+(v_|s_|ds_|buffer_|global_)", x)]
        c = sum(1 for x in ins if re.match(r"^\s+v_mov_b32_e32 v\d+, (-?[0-9.]+|0x[0-9a-fA-F]+)$", x))
        s = sum(1 for x in ins if re.match(r"^\s+v_mov_b32_e32 v\d+, s\d+$", x))
        valu = sum(1 for x in ins if re.match(r"^\s+v_(?!mfma|smfmac)", x))
        out.append((len(ins), m.group(1), c, s, valu))
out.sort(reverse=True)
for n, lab, c, s, v in out[:2]:
    print(f"loop {lab}: insts={n} valu={v} vmov_const={c} vmov_s={s}")
