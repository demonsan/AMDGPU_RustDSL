import sys, struct, collections
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
B, H, S, D = [int(x) for x in sys.argv[3:7]]
n = len(a) // 2
va = struct.unpack(f'<{n}H', a); vb = struct.unpack(f'<{n}H', b)
rows = set(); cnt = 0
zero_b = 0
for i in range(n):
    if va[i] != vb[i]:
        cnt += 1
        rows.add(i // D)
        if vb[i] == 0: zero_b += 1
print('differing words', cnt, 'of', n, '; of them ours==0:', zero_b)
per = collections.Counter(); blocks = collections.Counter(); inner = collections.Counter()
for r in rows:
    bb, hh, ss = r // (H * S), (r // S) % H, r % S
    per[(bb, hh)] += 1; blocks[ss // 256] += 1; inner[ss % 256] += 1
print('rows differing', len(rows), 'of', B * H * S)
print('(b,h) counts', sorted(per.items())[:40])
print('256-row block counts', sorted(blocks.items()))
print('in-block row counts (first)', sorted(inner.items())[:20])
