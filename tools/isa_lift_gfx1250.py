#!/usr/bin/env python3
"""gfx1250 front end of the exact lifter (tools/isa_lift.py; docs/exact-mode.md): the node's `llvm-objdump --mcpu=gfx1250` text of one kernel
-> the lifter's instruction list, then the same structured whole-kernel lift into DSL IR text (irv1).

What differs from gfx950 and is handled here:
- the input is normalised by tools/isa_parse_gfx1250.py: physical VGPR numbers through `s_set_vgpr_msb` (the encoded numbers are what the
  instruction stream holds; the MSB instructions are lifted as pinned `SetVgprMsb` statements), the tensor-data-mover triples decoded;
- branch operands are word offsets: their targets become labels `L<addr>`;
- block-scaled WMMAs whose second VOP3PX2 half the disassembler could not print are rebuilt from the raw words (operands from the encoding
  fields, physical registers from the current `s_set_vgpr_msb` state);
- mnemonics that are the gfx950 instruction under a new name map onto the gfx950 handlers (the gfx1250 emitter respells them back);
- the entry registers: v0 = the packed workitem ids, s[0:1] = the kernarg segment, s2.. = the preloaded kernarg words (`--preload N`), the
  workgroup ids are architected (TTMP9 / TTMP7) and not in SGPRs.

    isa_lift_gfx1250.py KERNEL.dis --kernel --exact --name K [--preload 20] [--out K.ir] [--report]  (other isa_lift.py options pass through)

The output encodes the input's order and registers: a working artifact of the session, never committed.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_lift  # noqa: E402
import isa_parse_gfx1250 as P  # noqa: E402

# gfx1250 spelling -> the gfx950 handler's (the inverse of the gfx1250 emitter's RENAMES; only same-semantics renames)
TO_950 = {
    'ds_load_b32': 'ds_read_b32', 'ds_load_b64': 'ds_read_b64', 'ds_load_b128': 'ds_read_b128', 'ds_store_b32': 'ds_write_b32',
    'ds_store_b64': 'ds_write_b64', 'ds_store_b128': 'ds_write_b128', 'v_add_nc_u32': 'v_add_u32', 'v_sub_nc_u32': 'v_sub_u32',
    's_add_co_u32': 's_add_u32', 's_add_co_ci_u32': 's_addc_u32', 's_sub_co_u32': 's_sub_u32', 's_load_b32': 's_load_dword',
    's_load_b64': 's_load_dwordx2', 'global_load_b128': 'global_load_dwordx4', 'global_load_b32': 'global_load_dword',
}
WAITS = {'s_wait_loadcnt': 'Load', 's_wait_storecnt': 'Store', 's_wait_dscnt': 'Ds', 's_wait_kmcnt': 'Km', 's_wait_xcnt': 'X',
         's_wait_asynccnt': 'Async', 's_wait_tensorcnt': 'Tensor'}
BRANCHES = ('s_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_vccz', 's_cbranch_vccnz', 's_cbranch_execz', 's_cbranch_execnz')


def decode_vop3px2(first, second, msb):
    """v_wmma_scale_* rebuilt from its two VOP3PX2 halves: `first` = the parsed `v_wmma_ld_scale_paired_b32` record (scale sources and the
    scale modifiers), `second` = the raw words of the matrix half. Physical VGPRs from the MSB state (2 bits each: src0 [1:0], src1 [3:2],
    src2 [5:4], dst [7:6])."""
    w0, w1 = second['words']
    op = (w0 >> 16) & 0x7F
    vdst = w0 & 0xFF
    src = [(w1 >> 0) & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF]
    hi = [(msb >> 0) & 3, (msb >> 2) & 3, (msb >> 4) & 3, (msb >> 6) & 3]
    if op != 0x08:
        raise ValueError(f'VOP3PX2 matrix half with op {op:#x}')
    def v(x, k, n):
        if x < 256:
            return '0' if x == 0x80 else f'inline{x}'
        b = x - 256 + 256 * hi[k]
        return f'v[{b}:{b + n - 1}]'
    d = 256 * hi[3] + vdst
    args = [f'v[{d}:{d + 15}]', v(src[0], 0, 16), v(src[1], 1, 8), v(src[2], 2, 16)] + first['args'][:2]
    # the modifiers (scale row, reuse hints) are bits of the scale half: as printed for it
    mods = list(first['mods'])
    return args, mods


def parse(path):
    """(sec, op, args) of the whole kernel, gfx1250 input."""
    recs = P.parse(open(path, encoding='utf-8', errors='replace').read())
    targets = set()
    for r in recs:
        if r['op'] in BRANCHES and r['args']:
            s = int(r['args'][0], 0)
            s = s - 65536 if s >= 32768 else s
            r['target'] = r['addr'] + 4 + 4 * s
            targets.add(r['target'])
    # SCC results a later branch / select of the same straight-line run reads (their writer must define the condition value)
    scc_w = ('s_add', 's_sub', 's_and', 's_or', 's_xor', 's_andn2', 's_orn2', 's_nand', 's_nor', 's_xnor', 's_lshl', 's_lshr', 's_ashr',
             's_bfe', 's_bitcmp', 's_min', 's_max', 's_abs', 's_not', 's_mul_hi', 's_bcnt', 's_ff')
    scc_r = ('s_cbranch_scc', 's_cselect', 's_add_co_ci', 's_sub_co_ci', 's_cmov')
    scc_live = set()
    for k, r in enumerate(recs):
        if not r['op'].startswith(scc_w) or r['op'].startswith(('s_cmp', 's_add_nc', 's_sub_nc')):
            continue
        for j in range(k + 1, len(recs)):
            o = recs[j]['op']
            if recs[j]['addr'] in targets:
                break
            if o.startswith(scc_r):
                scc_live.add(k)
                break
            if (o.startswith(scc_w) and not o.startswith(('s_add_nc', 's_sub_nc'))) or o.startswith(('s_cmp', 's_branch', 's_endpgm')):
                break
    out, sec, msb = [], 'entry', 0
    i = 0
    while i < len(recs):
        r = recs[i]
        if r['addr'] in targets:
            sec = f'L{r["addr"]:x}'
        op, args, mods = r['op'], list(r['args']), list(r['mods'])
        if op == 's_code_end':
            # the toolchain's padding after the code: kd "code_end_pad", not instructions of the kernel
            i += 1
            continue
        if op == 's_set_vgpr_msb':
            msb = int(args[0], 0) & 0xFF
        words = list(r['words'])
        if op == 'v_wmma_ld_scale_paired_b32' and i + 1 < len(recs) and recs[i + 1]['op'] == 'raw':
            args, mods = decode_vop3px2(r, recs[i + 1], msb)
            op = 'v_wmma_scale_f32_32x16x128_f4'
            words += recs[i + 1]['words']
            i += 1
        if op.startswith('v_wmma'):
            mods = mods + ['words:' + '/'.join(f'{w:#x}' for w in words)]
        elif op in BRANCHES:
            args = [f'L{r["target"]:x}'] if 'target' in r else args
        elif op in ('tensor_load_to_lds', 'tensor_store_from_lds'):
            mods = mods + ['raw:' + '/'.join(f'{w:#x}' for w in r['words'])]
            # the descriptor groups as SGPR ranges (4, 8, 4, 4 dwords): the liveness sees every word
            args = [f's[{int(x[1:])}:{int(x[1:]) + n - 1}]' for x, n in zip(args, (4, 8, 4, 4))]
        base = re.sub(r'_e(32|64)$', '', op)
        e64 = op.endswith('_e64')
        op = TO_950.get(base, base) + ('_e64' if e64 else '')
        # (branches carry their target label only: the CFG reads it)
        text = '' if op in BRANCHES else r.get('text', '')
        if text and '--raw-text' not in sys.argv:
            # the author's encoding, word for word (fields the assembler fills differently stay the author's), the text as a comment
            text = '.long ' + ', '.join(f'{w:#010x}' for w in r['words']) + ' // ' + text
        if i in scc_live:
            mods = mods + ['scc_live']
        out.append((sec, op, ', '.join(args) + (' ' + ' '.join(mods) if mods else '') + (' ||| ' + text if text else '')))
        i += 1
    return out


class Lifter(isa_lift.Lifter):
    def lift(self, op, args):
        text = ''
        if ' ||| ' in args:
            args, text = args.split(' ||| ', 1)
        forced = sys.argv[sys.argv.index('--raw-ops') + 1].split(',') if '--raw-ops' in sys.argv else []
        if text and any(op == f or op.startswith(f + '_e') for f in forced) and ' scc_live' not in args and self.raw(op, args, text):
            # (forced verbatim: a lifted form whose encoding the assembler fills differently)
            self.cov[op] += 1
            self.raws[op + ' (forced)'] += 1
            return True
        if ' scc_live' in args:
            args = args.replace(' scc_live', '')
            if text and self.raw(op, args, text, scc=True):
                self.cov[op] += 1
                self.raws[op + ' (SCC read by a branch / select)'] += 1
                return True
        mark, nv0, npins, st0 = len(self.lines), self.nv, len(self.pins), dict(self.state)
        cov0, un0 = dict(self.cov), dict(self.unmapped)
        try:
            ok = super().lift(op, args)
        except (TypeError, ValueError, KeyError, IndexError, StopIteration) as e:
            # an operand form a gfx950 handler does not know (m0, null, ttmp, a 64-bit SALU pair...)
            self.notes[f'operand form ({type(e).__name__}) in {op}'] += 1
            ok = False
        if ok:
            return True
        # not lifted: undo, then (exact mode, --raw) the instruction verbatim with its register dataflow declared
        del self.lines[mark:]
        del self.pins[npins:]
        self.views = {k: v for k, v in self.views.items() if v < nv0}
        self.state = st0
        self.cov, self.unmapped = isa_lift.collections.Counter(cov0), isa_lift.collections.Counter(un0)
        if '--raw' in sys.argv and text and self.raw(op, args, text):
            self.cov[op] += 1
            self.raws[op] += 1
            return True
        self.unmapped[op] += 1
        self.stmt(f'# UNMAPPED {op} {args}')
        return False

    raws = isa_lift.collections.Counter()

    def livein(self, f, base, n):
        # registers the kernel reads before any definition on some path (accumulators of a first pass, the other arm's values): an
        # undefined value pinned there, no instruction (exact: the author issues no initialisation)
        ty = 'I32' if n == 1 else isa_lift.FRAG.get(n, 'Frag16')
        if f == 's' and n not in (1,):
            ty = {2: 'Ptr', 4: 'Rsrc'}.get(n, ty)
        self.notes[f'live-in {f} register(s): ghost definition'] += 1
        return self.new(f, base, n, ty, f'Raw {{ text: "", reads: [], ty: {ty}, scalar: {"true" if f == "s" else "false"}, dwords: {n} }}', shell=True)

    # instructions without a register destination (the first operand is a source), and those that read their destination too
    NO_DST = ('s_cmp', 's_bitcmp', 's_setreg', 's_wait', 's_prefetch', 's_version', 'global_prefetch', 's_code_end', 's_nop', 'v_nop',
              's_barrier', 's_sleep', 's_setprio', 'ds_write', 'ds_store', 'tensor_', 's_cbranch', 's_branch', 's_endpgm', 'buffer_store',
              'global_store', 'v_cmp')
    RMW = ('s_bitset0', 's_bitset1', 's_addk', 's_mulk', 's_cmovk', 'v_swap')
    SCC_WRITERS = ('s_add', 's_sub', 's_bfe', 's_bitcmp', 's_and', 's_or', 's_xor', 's_lshl', 's_lshr', 's_ashr', 's_min', 's_max', 's_cmp')

    def raw(self, op, args, text, scc=False):
        t, _ = isa_lift.toks(args)
        regs = [isa_lift.reg(x) for x in t]
        has_dst = not op.startswith(self.NO_DST) and regs and regs[0] is not None
        reads = []
        for k, r in enumerate(regs):
            if r is None or (k == 0 and has_dst and not op.startswith(self.RMW)):
                continue
            f, a, n, _ = r
            reads.append(self.read(f, a, n))
        txt = text.replace('\\', '\\\\').replace('"', '\\"')
        rd = ', '.join(reads)
        if scc:
            # the value of the instruction is its SCC (read by the next branch / select); the register it writes keeps no lifted value
            # (later readers name the register through the lifter's state, the exact pin)
            c = self.new('scc', 0, 1, 'I1', f'Raw {{ text: "{txt}", reads: [{rd}], ty: I1, scalar: true, dwords: 1 }}')
            if has_dst:
                # the register the instruction writes too: a ghost definition (no instruction) ordered after it
                f, a, n, _ = regs[0]
                ty = 'I32' if n == 1 else {2: 'Ptr', 4: 'Rsrc', 8: 'Sgpr8'}.get(n) if f == 's' else isa_lift.FRAG.get(n)
                self.new(f, a, n, ty, f'Raw {{ text: "", reads: [Val({c})], ty: {ty}, scalar: {"true" if f == "s" else "false"}, dwords: {n} }}')
            return True
        if has_dst:
            f, a, n, _ = regs[0]
            ty = 'I32' if n == 1 else isa_lift.FRAG.get(n)
            if f == 's' and n > 1:
                # an SGPR tuple: the scalar tuple types (their words are read in place: PtrLo / PtrHi)
                ty = {2: 'Ptr', 4: 'Rsrc', 8: 'Sgpr8'}.get(n)
            if ty is None or (f == 's' and n not in (1, 2, 4, 8)):
                return False
            first = self.new(f, a, n, ty, f'Raw {{ text: "{txt}", reads: [{rd}], ty: {ty}, scalar: {"true" if f == "s" else "false"}, dwords: {n} }}')
            if op.startswith('v_swap') and len(regs) > 1 and regs[1] is not None:
                # the second destination: a ghost definition (empty text, no instruction) ordered after the swap
                f2, a2, n2, _ = regs[1]
                ty2 = 'I32' if n2 == 1 else isa_lift.FRAG.get(n2)
                self.new(f2, a2, n2, ty2, f'Raw {{ text: "", reads: [Val({first})], ty: {ty2}, scalar: false, dwords: {n2} }}')
        else:
            self.stmt(f'Raw {{ text: "{txt}", reads: [{rd}] }}')
        if op.startswith(self.SCC_WRITERS) and not op.startswith(('s_add_nc', 's_sub_nc')):
            # SCC now holds the raw instruction's result: no lifted compare stands for it any more
            self.state.pop(('scc', 0), None)
        return True

    def h_s_set_vgpr_msb(self, t, m, op):
        self.stmt(f'SetVgprMsb {{ imm: {int(t[0], 0)} }}')
        return True

    def _wait(self, t, m, op):
        self.stmt(f'WaitCounter {{ counter: {WAITS[op]}, count: {int(t[0], 0)} }}')
        return True

    def h_s_barrier_signal(self, t, m, op):
        self.stmt(f'BarrierSignal {{ id: {int(t[0], 0)} }}')
        return True

    def h_s_barrier_wait(self, t, m, op):
        i = int(t[0], 0)
        self.stmt(f'BarrierWait {{ id: {i - 65536 if i >= 32768 else i} }}')
        return True

    def _sgpr_words(self, base, n):
        return [self.coerce(self.read('s', base + i, 1), 'I32') for i in range(n)]

    def _tdm(self, t, m, op):
        g0 = isa_lift.reg(t[0])[1]
        g1 = isa_lift.reg(t[1])[1]
        raw = next(x for x in m if x.startswith('raw:'))[4:].split('/')
        th = next(int(x[3:]) for x in m if x.startswith('th:'))
        sc = next(int(x[6:]) for x in m if x.startswith('scope:'))
        w0 = self._sgpr_words(g0, 4)
        w1 = self._sgpr_words(g1, 8)
        a = self.nv
        self.nv += 1
        self.lines.append(f'%v{a}: Rsrc = RsrcCat {{ words: [{", ".join(w0)}] }}')
        self.vals[a] = ('s', g0, 4, 'Rsrc')
        self.pin('s', a, g0)
        b = self.nv
        self.nv += 1
        self.lines.append(f'%v{b}: Sgpr8 = SgprCat8 {{ words: [{", ".join(w1)}] }}')
        self.vals[b] = ('s', g1, 8, 'Sgpr8')
        self.pin('s', b, g1)
        kind = 'TensorLoadToLds' if op.startswith('tensor_load') else 'TensorStoreFromLds'
        words = ', '.join(str(int(x, 16)) for x in raw)
        self.stmt(f'{kind} {{ g0: Val({a}), g1: Val({b}), g2: None, g3: None, cpol: TdmCpol {{ th: {th}, scope: {sc}, nv: false }}, raw: Some([{words}]) }}')
        return True

    h_tensor_load_to_lds = _tdm
    h_tensor_store_from_lds = _tdm

    def _wmma(self, t, m, op, kind, scaled):
        f, d, nd, _ = isa_lift.reg(t[0])
        a = self.src(t[1])
        b = self.src(t[2])
        c = 'None' if t[3] == '0' else f'Some({self.src(t[3])})'
        sc = 'None'
        if scaled:
            sa, sb = self.src(t[4], 'I32'), self.src(t[5], 'I32')
            sc = f'Some(WmmaScale {{ sa: {sa}, sb: {sb}, b_row1: {"true" if any("MATRIX_SCALE_ROW1" in x for x in m) else "false"} }})'
        w = next((x[6:].split('/') for x in m if x.startswith('words:')), None)
        raw = 'None' if w is None or '--wmma-no-words' in sys.argv else 'Some([' + ', '.join(str(int(x, 16)) for x in w) + '])'
        self.dst(t[0], isa_lift.FRAG[nd], f'Wmma {{ kind: {kind}, a: {a}, b: {b}, c: {c}, scale: {sc}, raw: {raw} }}')
        return True

    def h_v_wmma_scale_f32_32x16x128_f4(self, t, m, op):
        return self._wmma(t, m, op, 'ScaleF4_32x16x128', True)

    def h_v_wmma_f32_16x16x64_fp8_fp8(self, t, m, op):
        return self._wmma(t, m, op, 'Fp8_16x16x64', False)

    def h_v_wmma_f32_16x16x128_fp8_fp8(self, t, m, op):
        return self._wmma(t, m, op, 'Fp8_16x16x128', False)


for _w in WAITS:
    setattr(Lifter, 'h_' + _w, Lifter._wait)


def _scmp(pred, ty):
    def h(self, t, m, op):
        self.new('scc', 0, 1, 'I1', f'Cmp {{ pred: {pred}, ty: {ty}, a: {self.src(t[0])}, b: {self.src(t[1])} }}')
        return True
    return h


for _p, _n in [('lt', 'Lt'), ('le', 'Le'), ('gt', 'Gt'), ('ge', 'Ge'), ('eq', 'Eq'), ('lg', 'Ne')]:
    for _t, _tn in [('u32', 'U32'), ('i32', 'I32')]:
        setattr(Lifter, f'h_s_cmp_{_p}_{_t}', _scmp(_n, _tn))


_defs_uses_950 = isa_lift.defs_uses
NO_DST_1250 = ('tensor_', 's_setreg', 's_wait', 's_prefetch', 's_version', 'global_prefetch', 's_set_vgpr_msb', 's_code_end', 's_bitcmp',
               's_sleep', 'v_nop')


def defs_uses(op, args):
    """isa_lift.defs_uses for the gfx1250 forms (the verbatim text and the decoded modifiers are not operands)."""
    args = args.split(' ||| ', 1)[0]
    args = ' '.join(w for w in args.split(' ') if not w.startswith(('raw:', 'th:', 'scope:', 'words:')) and w != 'scc_live')
    t, m = isa_lift.toks(args)
    rr = lambda tok: [(r[0], r[1] + i) for r in [isa_lift.reg(tok)] if r for i in range(r[2])]
    if op.startswith(NO_DST_1250):
        u = set()
        for x in t:
            u |= set(rr(x))
        return set(), u
    d, u = _defs_uses_950(op, args)
    if op.startswith(('s_bitset0', 's_bitset1', 's_cmovk', 's_mulk')):
        u |= d
    if op.startswith('v_swap') and len(t) > 1:
        d |= set(rr(t[1]))
        u |= set(rr(t[0]))
    return d, u


def entry_hw(L, live):
    """gfx1250 entry: v0 = the packed workitem ids (x [9:0], y [19:10], z [29:20]), s[0:1] = the kernarg segment, s2.. = preloaded kernarg
    dwords."""
    # (until the IR has the hardware-initialised forms: ghost definitions, no instruction)
    L.new('v', 0, 1, 'U32', 'Raw { text: "", reads: [], ty: U32, scalar: false, dwords: 1 }')
    L.new('s', 0, 2, 'Ptr', 'KernargSegment')
    n = int(sys.argv[sys.argv.index('--preload') + 1]) if '--preload' in sys.argv else 0
    for k in range(n):
        if ('s', 2 + k) in live:
            L.new('s', 2 + k, 1, 'I32', f'KernargPreload {{ dword: {k} }}')


def main():
    isa_lift.parse = parse
    isa_lift.Lifter = Lifter
    isa_lift.ENTRY_HW = entry_hw
    isa_lift.defs_uses = defs_uses
    isa_lift.RTY.update({'Wmma': None, 'SgprCat8': 'Sgpr8', 'KernargPreload': 'I32', 'WorkitemIdPacked': 'U32'})
    isa_lift.main()
    # the author's kernel-descriptor directives (--kd name=value,...): `kd "name" value` header lines after `exact`
    if '--kd' in sys.argv and '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
        kd = [x.split('=') for x in sys.argv[sys.argv.index('--kd') + 1].split(',')]
        txt = open(out).read().split('\n')
        k = txt.index('exact') + 1 if 'exact' in txt else 3
        txt[k:k] = [f'kd "{n}" {v}' for n, v in kd]
        open(out, 'w').write('\n'.join(txt))


if __name__ == '__main__':
    main()
