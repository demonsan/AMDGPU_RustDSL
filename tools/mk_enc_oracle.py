#!/usr/bin/env python3
"""The instruction oracle of the in-process assembler (docs/compile-time.md): every unique non-branch instruction line of the emitted assembly of the
manifest kernels (`amdc compile-batch --asm-dir DIR`) assembled by `llvm-mc -mcpu=gfx950 -show-encoding`; writes `asm<TAB>hex` lines for `amdc co-check`
and (with tools/gen_enc_tables.py) the opcode / format tables of `crates/amdgpu-target/src/gfx950_co/tables.rs`.

    python3 tools/mk_enc_oracle.py DIR [OUT.tsv]      (AMD_LLVM_BIN = the directory of llvm-mc)"""
import glob, os, re, subprocess, sys

d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/oracle.tsv'
mc = os.path.join(os.environ.get('AMD_LLVM_BIN', ''), 'llvm-mc')
BR = ('s_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_vccz', 's_cbranch_vccnz', 's_cbranch_execz', 's_cbranch_execnz')
lines = set()
for f in glob.glob(os.path.join(d, '*.s')):
    meta = False
    for line in open(f):
        l = line.strip()
        if l.startswith('.amdgpu_metadata'):
            meta = True
        if l.startswith('.end_amdgpu_metadata'):
            meta = False
            continue
        if meta or not l or l.startswith(('//', '.', '#')) or l.endswith(':'):
            continue
        l = l.split('//')[0].strip()
        if l and l.split()[0] not in BR:
            lines.add(l)
lines = sorted(lines)
r = subprocess.run([mc, '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx950', '-show-encoding'], input='.text\n' + '\n'.join(lines) + '\n', capture_output=True, text=True)
enc = [l for l in r.stdout.split('\n') if 'encoding:' in l]
if len(enc) != len(lines):
    sys.exit('llvm-mc rejected lines: ' + r.stderr[:600])
with open(out, 'w') as f:
    for l, e in zip(lines, enc):
        m = re.search(r'encoding: \[(.*)\]', e)
        f.write(l + '\t' + ''.join(x.strip()[2:] for x in m.group(1).split(',')) + '\n')
print(len(lines), 'unique instruction lines ->', out)