#!/usr/bin/env python3
"""Re-assemble a reference disassembly inside an exact build's .s wrapper (same kernel descriptor / metadata): isolates whether a
difference comes from the instruction stream or from the packaging.  usage: reasm_into.py REF.dis BUILD.s OUT.s KERNEL"""
import re, sys
ref, build, out, kname = sys.argv[1:5]
body = []
for l in open(ref):
    m = re.match(r'^[0-9a-f]+ <([^>]+)>:', l)
    if m:
        name = m.group(1)
        if not name.startswith('_Z'):
            body.append(f'{name}:')
        continue
    m = re.match(r'^\s+(\S.*?)\s*//', l)
    if m and not m.group(1).startswith('.'):
        body.append('    ' + m.group(1))
# the unnamed label of the kernel end
idx = max(i for i, x in enumerate(body) if x.strip() == 'label_skip_lse_2:')
body.insert(idx + 1, 'label_schedule_done:')
B = open(build).read().split('\n')
start = next(i for i, l in enumerate(B) if l.strip() == f'{kname}:')
end = next(i for i in range(start + 1, len(B)) if B[i].strip().startswith('s_endpgm'))
# keep everything after the build's s_endpgm (padding / descriptor / metadata)
open(out, 'w').write('\n'.join(B[:start + 1] + body + B[end + 1:]))
print('instructions', sum(1 for x in body if x.startswith('    ')))
