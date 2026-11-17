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
    's_load_b64': 's_load_dwordx2', 's_addk_co_i32': 's_addk_i32', 'ds_load_tr8_b64': 'ds_read_b64_tr_b8', 'v_max3_num_f32': 'v_max3_f32', 'v_max_num_f32': 'v_max_f32', 'buffer_store_b32': 'buffer_store_dword', 'v_subrev_nc_u32': 'v_subrev_u32', 'global_load_b128': 'global_load_dwordx4', 'global_load_b32': 'global_load_dword',
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
                # (a carry pair - s_add_co_u32 ... s_add_co_ci_u32 - is two DSL forms, the carry through SCC: not a branch condition)
                if not (r['op'] == 's_add_co_u32' and o == 's_add_co_ci_u32'):
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
        # a 64-bit add of a zero-extended 32-bit offset (`s_add_co_u32 lo, lo, x; s_add_co_ci_u32 hi, 0, hi`): one pointer add, both words
        if (op == 's_add_co_u32' and i + 1 < len(recs) and recs[i + 1]['op'] == 's_add_co_ci_u32' and (i + 1) not in scc_live
                and recs[i + 1]['addr'] not in targets):
            n = recs[i + 1]['args']
            lo, hi = isa_lift.reg(args[0]), isa_lift.reg(n[0])
            bhi = isa_lift.reg(n[2]) if len(n) > 2 else None
            # the base pair is the source whose register sits right below the carry's source (bhi); the other source is the offset
            r1, r2 = isa_lift.reg(args[1]), isa_lift.reg(args[2])
            blo, off_tok = None, None
            if bhi and r1 and r1[0] == 's' and r1[1] == bhi[1] - 1:
                blo, off_tok = r1, args[2]
            elif bhi and r2 and r2[0] == 's' and r2[1] == bhi[1] - 1:
                blo, off_tok = r2, args[1]
            if (lo and hi and blo and n[1] == '0' and hi[1] == lo[1] + 1 and lo[1] % 2 == 0 and blo[1] % 2 == 0):
                pw = '/'.join(f'{w:#x}' for w in r['words']) + ';' + '/'.join(f'{w:#x}' for w in recs[i + 1]['words'])
                out.append((sec, 's_ptradd_zext', f's[{lo[1]}:{lo[1] + 1}], s[{blo[1]}:{blo[1] + 1}], {off_tok} pwords:{pw}'))
                i += 2
                continue
        if op == 'v_wmma_ld_scale_paired_b32' and i + 1 < len(recs) and recs[i + 1]['op'] == 'raw':
            args, mods = decode_vop3px2(r, recs[i + 1], msb)
            op = 'v_wmma_scale_f32_32x16x128_f4'
            words += recs[i + 1]['words']
            i += 1
        wops = sys.argv[sys.argv.index('--words-ops') + 1].split(',') if '--words-ops' in sys.argv else []
        SYS = ('v_nop', 's_sleep', 's_version', 's_wait_idle', 's_prefetch_inst', 'global_prefetch_b8', 's_getreg_b32', 's_get_pc_i64')
        exec_w = op in ('s_mov_b32', 's_cselect_b32') and args and args[0] == 'exec_lo'
        if op.startswith(('v_wmma', 's_wait_alu', 's_setreg_imm32') + SYS) or exec_w or any(op == f or op.startswith(f + '_e') for f in wops):
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
        args = [f's{108 + int(x[4:])}' if re.fullmatch(r'ttmp\d+', x) else x for x in args]
        # wave32: the lane mask is one register; the lifter's mask model names it vcc / exec (the emitter writes vcc_lo / exec_lo)
        args = [{'vcc_lo': 'vcc', 'exec_lo': 'exec'}.get(x, x) for x in args]
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
        wops = sys.argv[sys.argv.index('--words-ops') + 1].split(',') if '--words-ops' in sys.argv else []
        if any(op == f or op.startswith(f + '_e') for f in wops) and ' words:' in args:
            # the DSL form with the author's encoding (fields an assembler fills differently, e.g. an unused VOP3 source)
            w = next(x for x in args.split(' ') if x.startswith('words:'))[6:].split('/')
            plain = ' '.join(x for x in args.split(' ') if not x.startswith('words:'))
            mark = len(self.lines)
            try:
                ok = super().lift(op, plain)
            except (TypeError, ValueError, KeyError, IndexError, StopIteration):
                ok = False
            # the words go right before the statement that IS the instruction: the last new value definition that is not a
            # zero-instruction view (bitcast, lane, concatenation, pointer word)
            at = None
            if ok:
                for k in range(len(self.lines) - 1, mark - 1, -1):
                    l = self.lines[k].strip()
                    if re.match(r'^%v\d+: \w+ = ', l) and not re.search(r'= (Bitcast|Frag\w*Lane|FragCat|PtrLo|PtrHi|PtrCat)', l):
                        at = k
                        break
            if ok and at is not None:
                # (before the instruction's encode_e64: that marks the statement right after it)
                if at > mark and self.lines[at - 1].strip() == 'EncodeE64':
                    at -= 1
                self.lines.insert(at, 'AuthorWords { words: [' + ', '.join(str(int(x, 16)) for x in w) + '] }')
                return True
            del self.lines[mark:]
            self.notes[f'{op}: no DSL form for the author words'] += 1
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

    def h_s_wait_alu(self, t, m, op):
        # the dependency-counter wait (depctr fields; the SOPP immediate)
        w = next(x[6:].split('/') for x in m if x.startswith('words:'))
        self.stmt(f'WaitAlu {{ imm: {int(w[0], 16) & 0xffff} }}')
        return True

    def h_s_setreg_imm32_b32(self, t, m, op):
        # the scheduling mode (WAVE_SCHED_MODE bits [1:0]) is a statement the exact checker models; other fields stay verbatim
        if 'HW_REG_WAVE_SCHED_MODE' in t[0] and t[0].replace(' ', '').endswith(',0,2)'):
            w = next(x[6:].split('/') for x in m if x.startswith('words:'))
            self.stmt(f'SetSchedMode {{ mode: {int(w[1], 16)} }}')
            return True
        w = next(x[6:].split('/') for x in m if x.startswith('words:'))
        return self._sys(m, 'SetRegImm', int(w[0], 16) & 0xffff, int(w[1], 16), [])

    def h_v_mul_u32_u24(self, t, m, op):
        return self.bin(t, 'Bin(MulU32U24, {a}, {b})', 'I32', 'I32')
    def h_v_mul_lo_u32(self, t, m, op): return self.bin(t, 'MulI32({a}, {b})', 'I32', 'I32')
    def h_v_max_i32(self, t, m, op): return self.bin(t, 'Bin(MaxI32, {a}, {b})', 'I32', 'I32')
    def h_v_sub_u32(self, t, m, op): return self.bin(t, 'Bin(SubU32, {a}, {b})', 'I32', 'I32')
    def h_v_cmp_gt_u32(self, t, m, op): return self.cmp64(t, 'Gt', 'U32')

    def _words(self, m):
        # (an instruction without operands: toks() takes its only modifier for an operand - h_s_wait_idle passes t + m)
        w = next(x[6:].split('/') for x in m if x.startswith('words:'))
        self.stmt('AuthorWords { words: [' + ', '.join(str(int(x, 16)) for x in w) + '] }')
        return [int(x, 16) for x in w]

    def _sys(self, m, op, imm, imm2, reads):
        self._words(m)
        self.stmt(f'Sys {{ op: {op}, imm: {imm}, imm2: {imm2}, reads: [{", ".join(reads)}] }}')
        return True

    def h_v_add_nc_i32(self, t, m, op): return self.bin(t, 'Bin(AddNcI32, {a}, {b})', 'I32', 'I32')
    def h_v_subrev_u32(self, t, m, op): return self.bin(t, 'Bin(SubrevU32, {a}, {b})', 'I32', 'I32')
    def h_v_rcp_iflag_f32(self, t, m, op):
        self.dst(t[0], 'F32', f'Un(RcpIflagF32, {self.src(t[1], "F32")})')
        return True

    def h_v_permlane16_swap_b32(self, t, m, op):
        a, b = self.src(t[0], 'I32'), self.src(t[1], 'I32')
        na = self.dst(t[0], 'U32', f'Permlane16SwapA {{ a: {a}, b: {b} }}')
        self.dst(t[1], 'U32', f'Permlane32SwapB {{ swap: Val({na}) }}')
        return True

    def h_v_add_nc_u64(self, t, m, op):
        a, b = self.src(t[1], 'Frag2'), self.src(t[2], 'Frag2')
        self.dst(t[0], 'Frag2', f'VAddU64 {{ a: {a}, b: {b} }}')
        return True

    def h_s_buffer_load_b32(self, t, m, op):
        off = next((int(x.split(':')[1], 0) for x in m if x.startswith('offset:')), 0)
        rs, so = self.src(t[1], 'Rsrc'), self.src(t[2], 'I32')
        self.dst(t[0], 'I32', f'SBufferLoad {{ rsrc: {rs}, soff: {so}, imm: {off} }}')
        return True

    def h_global_load_async_to_lds_b128(self, t, m, op):
        off = next((int(x.split(':')[1], 0) for x in m if x.startswith('offset:')), 0)
        self.stmt(f'LoadAsyncLds {{ lds: {self.src(t[0], "I32")}, vaddr: {self.src(t[1], "Frag2")}, imm: {off} }}')
        return True

    def h_v_cvt_pk_fp8_f32(self, t, m, op):
        # a 16-bit half write (true16 `.l` / `.h`): the register's other half is kept
        d = t[0]
        hi = d.endswith('.h')
        base = d[:-2] if d.endswith(('.l', '.h')) else d
        r = isa_lift.reg(base)
        # (a half the program never wrote: no old value - the other half is undefined)
        old = self.src(base, 'I32') if ('v', r[1]) in self.state else None
        a, b = self.src(t[1], 'F32'), self.src(t[2], 'F32')
        self.dst(base, 'U32', f'CvtPkFp8Half {{ old: {"Some(" + old + ")" if old else "None"}, a: {a}, b: {b}, hi: {"true" if hi else "false"} }}')
        return True

    def h_s_mov_b64(self, t, m, op):
        r0, r1 = isa_lift.reg(t[0]), isa_lift.reg(t[1]) if len(t) > 1 else None
        if r0 and r1 and r0[0] == 's' and r1[0] == 's' and r0[2] == 2 and r1[2] == 2:
            self.dst(t[0], 'Ptr', f'Salu {{ op: MovB64, a: {self.src(t[1], "Ptr")}, b: I32(0) }}')
            return True
        return super().h_s_mov_b64(t, m, op)

    def h_v_nop(self, t, m, op):
        return self._sys(list(t) + list(m), 'VNop', 0, 0, [])

    def h_s_sleep(self, t, m, op):
        return self._sys(m, 'Sleep', isa_lift.imm(t[0]) & 0xffff, 0, [])

    def _ds_store(self, t, m, ty):
        addr = self.lds_addr(['', t[0]], m)
        self.stmt(f'StoreLdsN {{ addr: {addr}, value: {self.src(t[1], "I32" if ty == "B32" else None)}, ty: {ty}, hi: false }}')
        return True

    def h_ds_write_b32(self, t, m, op): return self._ds_store(t, m, 'B32')
    def h_ds_write_b64(self, t, m, op): return self._ds_store(t, m, 'B64')
    def h_v_mul_hi_u32(self, t, m, op): return self.bin(t, 'Bin(MulHiU32, {a}, {b})', 'I32', 'I32')
    def h_v_cmp_le_u32(self, t, m, op): return self.cmp64(t, 'Le', 'U32')
    def h_v_cvt_f32_u32(self, t, m, op):
        self.dst(t[0], 'F32', f'Un(CvtF32U32, {self.src(t[1], "I32")})')
        return True
    def h_v_cvt_u32_f32(self, t, m, op):
        self.dst(t[0], 'U32', f'Un(CvtU32F32, {self.src(t[1], "F32")})')
        return True
    def h_v_pk_add_f32(self, t, m, op):
        a, b = self.src(t[1]), self.src(t[2])
        self.dst(t[0], 'Frag2', f'PkAddF32({a}, {b})')
        return True

    def h_s_version(self, t, m, op):
        w = next(x[6:].split('/') for x in m if x.startswith('words:'))
        return self._sys(m, 'Version', int(w[0], 16) & 0xffff, 0, [])

    def h_s_wait_idle(self, t, m, op):
        return self._sys(list(t) + list(m), 'WaitIdle', 0, 0, [])

    def h_s_prefetch_inst(self, t, m, op):
        return self._sys(m, 'PrefetchInst', isa_lift.imm(t[1]) & 0xffff, isa_lift.imm(t[3]) or 0, [self.src(t[0], 'Ptr'), self.src(t[2], 'I32')])

    def h_global_prefetch_b8(self, t, m, op):
        off = next((int(x.split(':')[1], 0) for x in m if x.startswith('offset:')), 0)
        return self._sys(m, 'PrefetchGlobal', off & 0xffff, 0, [self.src(t[0], 'I32'), self.src(t[1], 'Ptr')])

    def h_s_getreg_b32(self, t, m, op):
        w = self._words(m)
        self.dst(t[0], 'I32', f'SysVal {{ op: GetReg, imm: {w[0] & 0xffff} }}')
        return True

    def h_s_get_pc_i64(self, t, m, op):
        self._words(m)
        self.dst(t[0], 'Ptr', 'SysVal { op: GetPc, imm: 0 }')
        return True

    def h_s_mov_b32(self, t, m, op):
        # wave32 EXEC writes (the author's words: the shared lowering spells the 64-bit forms)
        if t[0] == 'exec':
            if t[1] == 'vcc':
                if ('vcc', 0) not in self.state:
                    return False
                self._words(m)
                self.stmt(f'ExecFromMask {{ mask: Val({self.state[("vcc", 0)][0]}) }}')
                return True
            k = isa_lift.imm(t[1])
            if k is not None:
                self._words(m)
                self.stmt('ExecAll' if (k & 0xffffffff) == 0xffffffff else f'ExecSet {{ mask: {k & 0xffffffff} }}')
                return True
            return False
        return super().h_s_mov_b32(t, m, op)

    def h_s_cselect_b32(self, t, m, op):
        if t[0] == 'exec':
            if ('scc', 0) not in self.state:
                return False
            c = self.state[('scc', 0)][0]
            a, b = self.src(t[1], 'I32'), self.src(t[2], 'I32')
            self._words(m)
            self.stmt(f'ExecSelect {{ cond: Val({c}), a: {a}, b: {b} }}')
            return True
        return super().h_s_cselect_b32(t, m, op)

    def h_s_ptradd_zext(self, t, m, op):
        # the author's pair (s_add_co_u32 + s_add_co_ci_u32, constant-first) as one pointer add; each instruction keeps its words
        pw = next(x for x in m if x.startswith('pwords:'))[7:].split(';')
        base, off = self.src(t[1], 'Ptr'), self.src(t[2], 'I32')
        for ws in pw:
            self.stmt('AuthorWords { words: [' + ', '.join(str(int(x, 16)) for x in ws.split('/')) + '] }')
        self.dst(t[0], 'Ptr', f'PtrAdd {{ base: {base}, byte_off: {off} }}')
        return True

    def pair_ok(self, a, b):
        # (the gfx950 lifter's s_add_u32 + s_addc_u32 pointer pairing probes registers with side effects; the gfx1250 front end pairs
        # them itself: s_ptradd_zext, or the AddCoU32 / AddCoCiU32 forms)
        return False

    def h_s_add_u32(self, t, m, op):
        # (the low word of a carry pair; m0 forms are the shared handler's)
        if t[0] == 'm0':
            return super().h_s_add_u32(t, m, op)
        ok = self._salu(t, 'AddCoU32')
        self.state.pop(('scc', 0), None)
        return ok

    def h_s_addc_u32(self, t, m, op):
        return self._salu(t, 'AddCoCiU32')

    def _salu(self, t, sop, ty='I32'):
        a, b = self.src(t[1], 'I32' if ty == 'I32' else None), self.src(t[2], 'I32' if ty == 'I32' else None)
        self.dst(t[0], ty, f'Salu {{ op: {sop}, a: {a}, b: {b} }}')
        return True

    def h_s_add_co_i32(self, t, m, op):
        return self._salu(t, 'AddCoI32')

    def h_s_sub_co_i32(self, t, m, op):
        return self._salu(t, 'SubCoI32')

    def h_s_bfe_u32(self, t, m, op):
        if t[1].startswith('ttmp'):
            return False
        return self._salu(t, 'BfeU32')

    def h_s_add_nc_u64(self, t, m, op):
        return self._salu(t, 'AddNcU64', 'Ptr')

    def _bitset(self, t, sop):
        # in place: the destination's old value is the first operand
        old, bit = self.src(t[0], 'I32'), self.src(t[1], 'I32')
        self.dst(t[0], 'I32', f'Salu {{ op: {sop}, a: {old}, b: {bit} }}')
        return True

    def h_s_bitset0_b32(self, t, m, op):
        return self._bitset(t, 'Bitset0')

    def h_s_bitset1_b32(self, t, m, op):
        return self._bitset(t, 'Bitset1')

    def h_v_swap_b32(self, t, m, op):
        # both registers in place: SwapA is the instruction (vA's new content), Permlane32SwapB names vB's
        a, b = self.src(t[0], 'I32'), self.src(t[1], 'I32')
        na = self.dst(t[0], 'U32', f'SwapA {{ a: {a}, b: {b} }}')
        self.dst(t[1], 'U32', f'Permlane32SwapB {{ swap: Val({na}) }}')
        return True

    def h_ds_write_b128(self, t, m, op):
        # ds_store_b128 vaddr, vdata[4] offset:N
        addr = self.lds_addr(['', t[0]], m)
        self.stmt(f'StoreLdsB128 {{ addr: {addr}, value: {self.src(t[1])} }}')
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
    args = ' '.join(w for w in args.split(' ') if not w.startswith(('raw:', 'th:', 'scope:', 'words:', 'pwords:')) and w != 'scc_live')
    # (a high-half write keeps the low half another write made: it reads the register; the low half comes first, from nothing)
    half = bool(re.search(r'^v\d+\.h\b', args.strip()))
    args = re.sub(r'(v\d+)\.[lh]\b', r'\1', args)
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
    if half:
        # a 16-bit write keeps the register's other half: the destination is read too
        u |= d
    if op.startswith('v_swap') and len(t) > 1:
        d |= set(rr(t[1]))
        u |= set(rr(t[0]))
    return d, u


def entry_hw(L, live):
    """gfx1250 entry: v0 = the packed workitem ids (x [9:0], y [19:10], z [29:20]), s[0:1] = the kernarg segment, s2.. = preloaded kernarg
    dwords."""
    # (until the IR has the hardware-initialised forms: ghost definitions, no instruction)
    L.new('v', 0, 1, 'U32', 'WorkitemIdPacked')
    L.new('s', 0, 2, 'Ptr', 'KernargSegment')
    n = int(sys.argv[sys.argv.index('--preload') + 1]) if '--preload' in sys.argv else 0
    for k in range(n):
        if ('s', 2 + k) in live:
            L.new('s', 2 + k, 1, 'I32', f'KernargPreload {{ dword: {k} }}')
    for r in range(108, 124):
        if ('s', r) in live:
            # a trap temporary (hardware-initialised: workgroup / cluster ids), read in place
            L.new('s', r, 1, 'I32', f'ReadTtmp {{ idx: {r - 108} }}')


def main():
    isa_lift.parse = parse
    isa_lift.Lifter = Lifter
    isa_lift.ENTRY_HW = entry_hw
    isa_lift.defs_uses = defs_uses
    isa_lift.RTY.update({'SwapA': 'U32', 'Wmma': None, 'SgprCat8': 'Sgpr8', 'KernargPreload': 'I32', 'ReadTtmp': 'I32', 'WorkitemIdPacked': 'U32'})
    isa_lift.main()
    # the kernel's argument layout (--args P,P,I,...: 8-byte pointers / 4-byte words in kernarg order): the code object's metadata lists every
    # argument (a launcher sizes and copies the kernarg buffer by it), also those only the preload reads
    if '--args' in sys.argv and '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
        txt = [l for l in open(out).read().split('\n') if not l.startswith('arg Arg ')]
        lines, off = [], 0
        for k in sys.argv[sys.argv.index('--args') + 1].split(','):
            ty, sz = ('Ptr', 8) if k == 'P' else ('I32', 4)
            lines.append(f'arg Arg {{ name: "ka_{off:x}", ty: {ty}, kernarg_offset: {off}, size: {sz} }}')
            off += sz
        k = next(i for i, l in enumerate(txt) if l.startswith('var_tys')) + 1
        txt[k:k] = lines
        open(out, 'w').write('\n'.join(txt))
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
