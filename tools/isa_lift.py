#!/usr/bin/env python3
"""gfx950 disassembly -> DSL IR text (irv1) lifter (docs/exact-mode.md, route B of the 1:1 replication).

Generic: it knows instruction forms, not any kernel. Every instruction of the chosen straight-line sections becomes a DSL statement (a real
DSL op wherever one with a fixed one-instruction expansion exists), every defined value is pinned to the physical registers the input
used (`reg_plan PinV`), sub-tuple reads become zero-instruction views (`Frag*Lane` / `FragSlice`) and tuples assembled from separately
defined parts become `FragCat`, pinned at the same registers. Live-in registers are defined by a small shell before the lifted code.
Instructions without a DSL form are counted and listed (they would need an asm window).

usage: isa_lift.py FILE.dis --sections a[,b..] [--name K] [--out F.ir] [--report]

The output is a working artifact of the session (it encodes the input's order and registers): never commit it.
"""
import re, sys, collections

REG = re.compile(r'^(-?)([vs])(?:\[(\d+):(\d+)\]|(\d+))$')


def parse(path):
    sec = 'entry'
    out = []
    for line in open(path):
        m = re.match(r'^[0-9a-f]+ <([^>]+)>:', line)
        if m:
            sec = m.group(1)
            continue
        s = re.sub(r'\s*//.*$', '', line).strip()
        if not s or s.startswith('Disassembly') or s.endswith(':') or '\tfile' in line or s.startswith('.'):
            continue
        parts = s.split(None, 1)
        out.append((sec, parts[0], parts[1] if len(parts) > 1 else ''))
    return out


def toks(args):
    # split on commas outside brackets, keep modifiers (op_sel..., offset:N, offen, lds) as separate words
    t, depth, cur = [], 0, ''
    for ch in args:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            t.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        t.append(cur.strip())
    # the last operand may carry trailing modifiers separated by spaces
    mods = []
    if t:
        words = t[-1].split()
        t[-1] = words[0]
        mods = words[1:]
    return t, mods


def reg(tok):
    m = REG.match(tok)
    if not m:
        return None
    neg, f = m.group(1) == '-', m.group(2)
    if m.group(3) is not None:
        a, b = int(m.group(3)), int(m.group(4))
    else:
        a = b = int(m.group(5))
    return f, a, b - a + 1, neg


def imm(tok):
    try:
        return int(tok, 0)
    except ValueError:
        try:
            return int(tok, 16)
        except ValueError:
            return None


FRAG = {1: 'U32', 2: 'Frag2', 4: 'Frag4', 8: 'Frag8', 16: 'Frag16'}

# result types of the DSL ops the lifter emits (Op::result_ty in amdgpu-dsl)
RTY = {'LoadScalar': 'U32', 'AddK': 'I32', 'KernargSegment': 'Ptr', 'Permlane32SwapA': 'U32', 'Permlane32SwapB': 'U32', 'PtrLo': 'I32', 'PtrHi': 'I32', 'PtrCat': 'Ptr', 'RsrcCat': 'Rsrc', 'WorkitemId': 'U32', 'WorkgroupIdX': 'U32', 'WorkgroupIdY': 'U32', 'WorkgroupIdZ': 'U32', 'Undef': None, 'ConstI32': 'I32', 'ConstF32': 'F32', 'KernargI32': 'I32', 'KernargPtr': 'Ptr', 'BufferRsrc': 'Rsrc', 'MovV': None, 'MovS': None, 'NegF32': 'F32', 'MfmaBf16Wide': 'Frag16', 'BufferRsrcDyn': 'Rsrc', 'AddI32': 'I32', 'MulI32': 'I32', 'ShlI32': 'I32', 'LshrI32': 'I32',
       'AndI32': 'I32', 'MinI32': 'I32', 'XorI32': 'I32', 'AddF32': 'F32', 'SubF32': 'F32', 'MulF32': 'F32', 'MaxF32': 'F32', 'FmaF32': 'F32', 'Max3F32': 'F32',
       'Exp2F32': 'F32', 'Log2F32': 'F32', 'RcpF32': 'F32', 'Cmp': 'I1', 'NotI1': 'I1', 'ReadFirstLane': 'U32', 'Frag2Lane': 'U32', 'Frag4Lane': 'U32',
       'Frag16Lane': 'U32', 'CvtPknormPair': 'I32', 'CvtP8x4FromU16': 'U32', 'CvtPkBf16F32': 'U32', 'PkMulF32': 'Frag2', 'PkFmaF32': 'Frag2',
       'LoadLdsB32': 'U32', 'KernargF32': 'F32', 'PtrAdd': 'Ptr', 'BufferLoad': None, 'LoadGlobalN': 'I32', 'LoadLdsB128': 'Frag4', 'LoadLdsTrB8': 'Frag2', 'SmfmacFp8_16x16x128': 'Frag4', 'Mfma': 'Frag16', 'FragZero': None,
       'FragCat': None, 'FragSlice': None, 'Select': None, 'Bin': 'I32', 'Tri': 'I32'}


def rty(rhs, default):
    op = re.match(r'(\w+)', rhs).group(1)
    t = RTY.get(op, default)
    if t is None:
        if op == 'FragSlice':
            n = int(re.search(r'n: (\d+)', rhs).group(1))
            return {1: 'I32', 2: 'Frag2', 4: 'Frag4'}.get(n, 'Frag8')
        if op in ('Select', 'MovV', 'MovS'):
            return re.search(r'ty: (\w+)', rhs).group(1)
        if op == 'BufferLoad':
            return {'1': 'F32', '2': 'Frag2'}.get(re.search(r'dwords: (\d+)', rhs).group(1), 'Frag4')
        return default
    return t


class Unlift(Exception):
    pass


class Lifter:
    def __init__(self):
        self.lines = []          # body statements
        self.pins = []           # reg_plan lines
        self.shell = []          # live-in definitions
        self.shell_pins = []     # their pins
        self.nv = 0
        self.vals = {}           # id -> (file, base, n, ty)
        self.state = {}          # (file, reg) -> (value id, offset)
        self.views = {}          # (id, start, n) -> view id
        self.cov = collections.Counter()
        self.unmapped = collections.Counter()
        self.notes = collections.Counter()
        self.varof = {}          # pseudo value id -> scope variable id (registers carried across blocks)
        self.lgkm = []           # --fix-waits: outstanding LGKM ops (op, defined registers) of the current block, oldest first
        self.added_waits = []
        self.added_nops = []
        self.varreg = {}         # scope variable id -> (file, register)

    def pin(self, f, v, base, shell=False):
        if f in ('v', 's'):
            (self.shell_pins if shell else self.pins).append(f'reg_plan {"PinV" if f == "v" else "PinS"} {{ r: Val({v}), base: {base} }}')

    def new(self, f, base, n, ty, rhs, shell=False):
        v = self.nv
        self.nv += 1
        ty = rty(rhs, ty)
        (self.shell if shell else self.lines).append(f'%v{v}: {ty} = {rhs}')
        self.vals[v] = (f, base, n, ty)
        # live-in (shell) pins survive the rollback of an instruction that could not be lifted (the shell lines do too)
        pins = self.shell_pins if shell else self.pins
        if f == 'v':
            pins.append(f'reg_plan PinV {{ r: Val({v}), base: {base} }}')
        elif f == 's':
            pins.append(f'reg_plan PinS {{ r: Val({v}), base: {base} }}')
        elif f == 'vcc':
            # an I1 in VCC (exact-mode convention: PinS base 106)
            pins.append(f'reg_plan PinS {{ r: Val({v}), base: 106 }}')
        elif f == 'scc':
            # an I1 in SCC (exact-mode convention: PinS base 253, the operand encoding of SCC)
            pins.append(f'reg_plan PinS {{ r: Val({v}), base: 253 }}')
        for i in range(n):
            self.state[(f, base + i)] = (v, i)
        return v

    def livein(self, f, base, n):
        if f == 's':
            if n == 4:
                p = self.nv
                self.nv += 1
                self.shell.append(f'%v{p}: Ptr = KernargPtr {{ offset: 0 }}')
                self.shell.append('WaitLds { count: 0 }')
                self.shell_pins.append(f'reg_plan PinS {{ r: Val({p}), base: {base} }}')
                return self.new('s', base, n, 'Rsrc', f'BufferRsrc {{ base: Val({p}), num_records: 4294967295 }}', shell=True)
            return self.new('s', base, n, 'U32' if n == 1 else 'Ptr', 'KernargI32 { offset: 0 }' if n == 1 else 'KernargPtr { offset: 0 }', shell=True)
        # authored per-dword moves of 0 (exact mode: a pinned constant is a v_mov at its statement), assembled without copies
        if n == 1:
            return self.new('v', base, 1, 'I32', 'MovV { ty: I32, a: I32(0) }', shell=True)
        parts = []
        for d in range(n):
            q = self.nv
            self.nv += 1
            self.shell.append(f'%v{q}: I32 = MovV {{ ty: I32, a: I32(0) }}')
            self.shell_pins.append(f'reg_plan PinV {{ r: Val({q}), base: {base + d} }}')
            parts.append(f'Val({q})')
        return self.new('v', base, n, FRAG[n], 'FragCat { parts: [' + ', '.join(parts) + f'], dwords: {n} }}', shell=True)

    def read(self, f, base, n):
        # the value(s) currently in registers [base, base + n)
        if any((f, base + i) not in self.state for i in range(n)):
            # live-in: define the missing dwords as one shell value when all are missing, else per dword
            if all((f, base + i) not in self.state for i in range(n)):
                self.livein(f, base, n)
            else:
                for i in range(n):
                    if (f, base + i) not in self.state:
                        self.livein(f, base + i, 1)
        v0, o0 = self.state[(f, base)]
        _, vb, vn, vty = self.vals[v0]
        whole = o0 == 0 and vn == n and all(self.state[(f, base + i)] == (v0, i) for i in range(n))
        if v0 in self.varof and whole:
            # a register (tuple) carried across a block boundary: its scope variable
            return f'Var(VarId({self.varof[v0]}))'
        if whole:
            return f'Val({v0})'
        if all(self.state[(f, base + i)] == (v0, o0 + i) for i in range(n)):
            if vty == 'Ptr' and n == 1:
                # a word of a pointer pair, read in place (s)
                src = f'Var(VarId({self.varof[v0]}))' if v0 in self.varof else f'Val({v0})'
                key = ('ptrword', v0, o0)
                if key not in self.views:
                    w = self.nv
                    self.nv += 1
                    self.lines.append(f'%v{w}: I32 = {"PtrHi" if o0 else "PtrLo"}({src})')
                    self.vals[w] = (f, base, 1, 'I32')
                    self.pin(f, w, base)
                    self.views[key] = w
                return f'Val({self.views[key]})'
            if vty in ('Ptr', 'Rsrc', 'I1'):
                raise Unlift(f'part of a {vty} value')
            key = (v0, o0, n)
            if key not in self.views:
                if n == 1:
                    rhs = f'{FRAG[vn]}Lane {{ frag: Val({v0}), index: {o0} }}' if vn in (2, 4, 16) else f'FragSlice {{ frag: Val({v0}), start: {o0}, n: 1 }}'
                else:
                    rhs = f'FragSlice {{ frag: Val({v0}), start: {o0}, n: {n} }}'
                self.notes['view'] += 1
                v = self.nv
                self.nv += 1
                self.lines.append(f'%v{v}: {rty(rhs, FRAG.get(n, "U32"))} = {rhs}')
                self.vals[v] = (f, base, n, FRAG.get(n, 'U32'))
                self.pin(f, v, base)
                self.views[key] = v
            return f'Val({self.views[key]})'
        if f == 's':
            # SGPR words assembled into a pointer / descriptor (s): PtrCat / RsrcCat of per-word reads
            if n not in (2, 4):
                raise Unlift(f's[{base}:{base + n - 1}] assembled from parts')
            words = [self.coerce(self.read(f, base + i, 1), 'I32') for i in range(n)]
            v = self.nv
            self.nv += 1
            if n == 2:
                self.lines.append(f'%v{v}: Ptr = PtrCat {{ lo: {words[0]}, hi: {words[1]} }}')
                self.vals[v] = (f, base, 2, 'Ptr')
            else:
                self.lines.append(f'%v{v}: Rsrc = RsrcCat {{ words: [{", ".join(words)}] }}')
                self.vals[v] = (f, base, 4, 'Rsrc')
            self.pin(f, v, base)
            self.notes['sgpr words assembled (PtrCat / RsrcCat)'] += 1
            if '--keep-tuples' in sys.argv and not any((f, base + i) in getattr(self, 'single', ()) for i in range(n)):
                # the registers now hold that tuple value (so a block end hands it on as one Ptr / Rsrc variable)
                for i in range(n):
                    self.state[(f, base + i)] = (v, i)
            return f'Val({v})'
        # assembled from several values: FragCat of the parts (each part a view when needed)
        parts, i = [], 0
        while i < n:
            vi, oi = self.state[(f, base + i)]
            j = i
            while j + 1 < n and self.state[(f, base + j + 1)] == (vi, oi + j + 1 - i):
                j += 1
            parts.append(self.read(f, base + i, j - i + 1))
            i = j + 1
        self.notes['cat'] += 1
        v = self.nv
        self.nv += 1
        self.lines.append(f'%v{v}: {FRAG[n]} = FragCat {{ parts: [{", ".join(parts)}], dwords: {n} }}')
        self.vals[v] = (f, base, n, FRAG[n])
        self.pin(f, v, base)
        return f'Val({v})'

    def coerce(self, x, want):
        # a 32-bit value used as another 32-bit type: a zero-instruction bitcast pinned to the same register
        mv = re.match(r'Var\(VarId\((\d+)\)\)$', x)
        if mv and want == 'F32':
            k = int(mv.group(1))
            f, base = self.varreg[k]
            key = ('vcast', k, want)
            if key not in self.views:
                w = self.nv
                self.nv += 1
                self.lines.append(f'%v{w}: F32 = BitcastF32({x})')
                self.vals[w] = (f, base, 1, 'F32')
                self.pins.append(f'reg_plan {"PinV" if f == "v" else "PinS"} {{ r: Val({w}), base: {base} }}')
                self.views[key] = w
            return f'Val({self.views[key]})'
        m = re.match(r'Val\((\d+)\)$', x)
        if not m or want is None:
            return x
        v = int(m.group(1))
        f, base, n, ty = self.vals[v]
        if n != 1 or ty == want or (want in ('I32', 'I32s') and ty == 'I32'):
            return x
        if want == 'F32' and ty in ('U32', 'I32'):
            op = 'BitcastF32'
        elif want in ('I32', 'U32', 'I32s') and ty == 'F32':
            op = 'BitcastI32'
        elif want == 'I32s' and ty == 'U32' and f == 'v':
            # an exact-typed operand (select arms): a zero-instruction view of the VGPR
            op = 'BitcastI32'
        else:
            # I32 / U32 are interchangeable integer types elsewhere: no view
            return x
        key = ('cast', v, 'I32' if want == 'I32s' else want)
        if key not in self.views:
            w = self.nv
            self.nv += 1
            rt = 'F32' if op == 'BitcastF32' else 'I32'
            self.lines.append(f'%v{w}: {rt} = {op}(Val({v}))')
            self.vals[w] = (f, base, 1, rt)
            self.pin(f, w, base)
            self.views[key] = w
            self.notes['bitcast view'] += 1
        return f'Val({self.views[key]})'

    def rsrc(self, tok):
        # a buffer descriptor operand: an Rsrc value, or four SGPRs the code assembled itself -> BufferRsrcDyn over its base / size words
        f, a, n, _ = reg(tok)
        x = self.read(f, a, n)
        if x.startswith('Var('):
            return x
        v = int(x[4:-1])
        if self.vals[v][3] == 'Rsrc':
            return x
        key = ('rsrc', a, tuple(self.state[(f, a + i)] for i in range(4)))
        if key not in self.views:
            base = self.read(f, a, 2)
            if self.vals[int(base[4:-1])][3] != 'Ptr':
                raise Unlift('descriptor base is not a pointer value')
            size = self.coerce(self.read(f, a + 2, 1), 'I32')
            w = self.nv
            self.nv += 1
            self.lines.append(f'%v{w}: Rsrc = BufferRsrcDyn {{ base: {base}, num_records: {size} }}')
            self.vals[w] = (f, a, 4, 'Rsrc')
            self.views[key] = w
            self.notes['descriptor assembled in SGPRs -> BufferRsrcDyn'] += 1
        return f'Val({self.views[key]})'
    def src(self, tok, want=None):
        r = reg(tok)
        if r:
            f, a, n, neg = r
            x = self.read(f, a, n)
            x = self.coerce(x, want)
            if neg:
                w = self.nv
                self.nv += 1
                self.lines.append(f'%v{w}: F32 = NegF32({self.coerce(x, "F32")})')
                self.notes['neg-modifier -> NegF32'] += 1
                return f'Val({w})'
            return x
        i = imm(tok)
        if i is not None and want == 'F32':
            import struct
            return f'F32({struct.unpack("<f", struct.pack("<I", i & 0xffffffff))[0]!r})'
        if i is not None:
            if -16 <= i <= 64:
                return f'I32({i})'
            return f'I32({i if i < 2**31 else i - 2**32})'
        try:
            x = float(tok)
            if want in ('I32', 'U32', 'I32s'):
                import struct
                return f'I32({struct.unpack("<i", struct.pack("<f", x))[0]})'
            return f'F32({x!r})'
        except ValueError:
            pass
        raise Unlift(f'operand {tok}')

    def stmt(self, s):
        self.lines.append(s)

    # ---- per-instruction handlers: return True when lifted ----
    def lift(self, op, args):
        t, mods = toks(args)
        base = re.sub(r'_e(32|64)$', '', op)
        h = getattr(self, 'h_' + base, None)
        if h is None:
            self.unmapped[op] += 1
            self.stmt(f'# UNMAPPED {op} {args}')
            return False
        mark, nv0, npins, st0 = len(self.lines), self.nv, len(self.pins), dict(self.state)
        try:
            ok = h(t, mods, op)
        except Unlift as e:
            del self.lines[mark:]
            del self.pins[npins:]
            self.views = {k: v for k, v in self.views.items() if v < nv0}
            self.state = st0
            self.notes[f'unliftable {e}'] += 1
            ok = False
        if ok:
            self.cov[op] += 1
        else:
            self.unmapped[op] += 1
            self.stmt(f'# UNMAPPED {op} {args}')
        return ok

    def dst(self, tok, ty, rhs):
        f, a, n, _ = reg(tok)
        if getattr(self, 'cur_e64', False):
            self.stmt('EncodeE64')
            self.cur_e64 = False
        return self.new(f, a, n, ty, rhs)


    def bin(self, t, rhs_fmt, ty='U32', want=None):
        a, b = self.src(t[1], want), self.src(t[2], want)
        if a is None or b is None:
            return False
        self.dst(t[0], ty, rhs_fmt.format(a=a, b=b))
        return True

    # VALU
    def h_v_add_u32(self, t, m, op): return self.bin(t, 'AddI32({a}, {b})', 'I32', 'I32')
    def h_v_and_b32(self, t, m, op): return self.bin(t, 'AndI32({a}, {b})', 'I32', 'I32')
    def h_v_sub_f32(self, t, m, op): return self.bin(t, 'SubF32({a}, {b})', 'F32', 'F32')
    def h_v_add_f32(self, t, m, op): return self.bin(t, 'AddF32({a}, {b})', 'F32', 'F32')
    def h_v_mul_f32(self, t, m, op): return self.bin(t, 'MulF32({a}, {b})', 'F32', 'F32')
    def h_v_max_f32(self, t, m, op): return self.bin(t, 'MaxF32({a}, {b})', 'F32', 'F32')
    def h_v_mul_i32_i24(self, t, m, op): return self.bin(t, 'Bin(MulI32I24, {a}, {b})', 'I32', 'I32')
    def h_v_max_u32(self, t, m, op): return self.bin(t, 'Bin(MaxU32, {a}, {b})', 'I32', 'I32')
    def h_v_mbcnt_lo_u32_b32(self, t, m, op): return self.bin(t, 'Bin(MbcntLo, {a}, {b})', 'I32', 'I32')
    def h_v_mbcnt_hi_u32_b32(self, t, m, op): return self.bin(t, 'Bin(MbcntHi, {a}, {b})', 'I32', 'I32')
    def h_v_lshrrev_b32(self, t, m, op): return self.bin(t, 'LshrI32({b}, {a})', 'I32', 'I32')
    def h_v_lshlrev_b32(self, t, m, op): return self.bin(t, 'ShlI32({b}, {a})', 'I32', 'I32')
    def h_v_cvt_pknorm_u16_f32(self, t, m, op): return self.bin(t, 'CvtPknormPair({a}, {b})', 'I32', 'F32')
    def h_v_cvt_pk_bf16_f32(self, t, m, op): return self.bin(t, 'CvtPkBf16F32({a}, {b})', 'U32', 'F32')

    def h_v_max3_f32(self, t, m, op):
        a, b, c = self.src(t[1], 'F32'), self.src(t[2], 'F32'), self.src(t[3], 'F32')
        self.dst(t[0], 'F32', f'Max3F32({a}, {b}, {c})')
        return True

    def h_v_fma_f32(self, t, m, op):
        a, b, c = self.src(t[1], 'F32'), self.src(t[2], 'F32'), self.src(t[3], 'F32')
        self.dst(t[0], 'F32', f'FmaF32({a}, {b}, {c})')
        return True

    def h_v_add3_u32(self, t, m, op):
        a, b, c = self.src(t[1]), self.src(t[2]), self.src(t[3])
        self.dst(t[0], 'U32', f'Tri(Add3U32, {a}, {b}, {c})')
        return True

    def h_v_exp_f32(self, t, m, op):
        self.dst(t[0], 'F32', f'Exp2F32({self.src(t[1], "F32")})')
        return True

    def h_v_log_f32(self, t, m, op):
        self.dst(t[0], 'F32', f'Log2F32({self.src(t[1], "F32")})')
        return True

    def h_v_rcp_f32(self, t, m, op):
        self.dst(t[0], 'F32', f'RcpF32({self.src(t[1], "F32")})')
        return True

    def h_v_readfirstlane_b32(self, t, m, op):
        self.dst(t[0], 'U32', f'ReadFirstLane({self.src(t[1], "I32")})')
        return True

    def h_v_mov_b32(self, t, m, op):
        i = imm(t[1])
        if i is not None:
            self.dst(t[0], 'I32', f'MovV {{ ty: I32, a: I32({i if i < 2**31 else i - 2**32}) }}')
            return True
        try:
            x = float(t[1])
            self.dst(t[0], 'F32', f'MovV {{ ty: F32, a: F32({x!r}) }}')
            return True
        except ValueError:
            pass
        # a register copy: the DSL has no "mov" op (copies are the allocator's); count it, express as a view of the source value
        s = self.src(t[1], 'I32')
        self.dst(t[0], 'I32', f'MovV {{ ty: I32, a: {s} }}')
        return True

    def h_v_pk_mul_f32(self, t, m, op): return self.bin(t, 'PkMulF32({a}, {b})', 'Frag2')

    def h_v_pk_fma_f32(self, t, m, op):
        a, b, c = self.src(t[1]), self.src(t[2]), self.src(t[3])
        self.dst(t[0], 'Frag2', f'PkFmaF32({a}, {b}, {c})')
        return True

    def h_v_permlane32_swap_b32(self, t, m, op):
        # swaps two registers in place: the A form is the instruction (new content of vA), the B form names vB's new content
        a, b = self.src(t[0], 'I32'), self.src(t[1], 'I32')
        na = self.dst(t[0], 'U32', f'Permlane32SwapA {{ a: {a}, b: {b} }}')
        self.dst(t[1], 'U32', f'Permlane32SwapB {{ swap: Val({na}) }}')
        return True

    def h_v_cvt_scalef32_pk_fp8_bf16(self, t, m, op):
        # the lo / hi pair is one DSL op (CvtP8x4FromU16 = 2 instructions); the lifter pairs them in `run`
        return False

    def h_v_cmp_nge_f32(self, t, m, op):
        if t[0] != 'vcc':
            return False
        # nge = NOT (a >= b) (true for unordered): the negated compare, one v_cmp_nge (both values in VCC)
        a, b = self.src(t[1], 'F32'), self.src(t[2], 'F32')
        c = self.new('vcc', 0, 1, 'I1', f'Cmp {{ pred: Ge, ty: F32, a: {a}, b: {b} }}')
        self.new('vcc', 0, 1, 'I1', f'NotI1(Val({c}))')
        self.notes['cmp-vcc nge = NotI1(Cmp Ge)'] += 1
        return True

    def h_v_cmp_lt_i32(self, t, m, op):
        if t[0] != 'vcc':
            return False
        a, b = self.src(t[1]), self.src(t[2])
        self.new('vcc', 0, 1, 'I1', f'Cmp {{ pred: Lt, ty: I32, a: {a}, b: {b} }}')
        return True

    # LDS
    def lds_addr(self, t, m):
        a = self.src(t[1])
        off = 0
        for x in m:
            if x.startswith('offset:'):
                off = int(x.split(':')[1], 0)
        if off:
            v = self.nv
            self.nv += 1
            self.lines.append(f'%v{v}: I32 = AddI32({a}, I32({off}))')
            self.notes['lds-offset folded'] += 1
            return f'Val({v})'
        return a

    def h_ds_read_b64_tr_b8(self, t, m, op):
        self.dst(t[0], 'Frag2', f'LoadLdsTrB8 {{ addr: {self.lds_addr(t, m)} }}')
        return True

    def h_ds_read_b128(self, t, m, op):
        self.dst(t[0], 'Frag4', f'LoadLdsB128 {{ addr: {self.lds_addr(t, m)} }}')
        return True

    def h_ds_read_b32(self, t, m, op):
        self.dst(t[0], 'U32', f'LoadLdsB32 {{ addr: {self.lds_addr(t, m)} }}')
        return True

    # MFMA
    def mfma(self, t, m, scaled, shape, ta, tb):
        a, b = self.src(t[1]), self.src(t[2])
        c = 'None' if t[3] == '0' else f'Some({self.src(t[3])})'
        sc = 'None'
        if scaled:
            opsel = 0
            for x in m:
                # the E8M0 byte of each scale dword: byte = op_sel[k] + 2 * op_sel_hi[k] (k = 0: scale_a, 1: scale_b); MmaScale.opsel holds
                # the byte of scale_a in bits 0..1 and of scale_b in bits 4..5
                if x.startswith('op_sel_hi:'):
                    hi = [int(y) for y in x.split(':')[1].strip('[]').split(',')]
                    opsel |= (hi[0] << 1) | (hi[1] << 5)
                elif x.startswith('op_sel:'):
                    lo = [int(y) for y in x.split(':')[1].strip('[]').split(',')]
                    opsel |= lo[0] | (lo[1] << 4)
            sc = f'Some(MmaScale {{ sa: {self.src(t[4])}, sb: {self.src(t[5])}, opsel: {opsel} }})'
        kind = f'MmaKind {{ shape: {shape}, a: {ta}, b: {tb}, scaled: {"true" if scaled else "false"} }}'
        self.dst(t[0], 'Frag16', f'Mfma {{ kind: {kind}, a: {a}, b: {b}, c: {c}, scale: {sc} }}')
        return True

    def h_v_mfma_scale_f32_32x32x64_f8f6f4(self, t, m, op): return self.mfma(t, m, True, 'M32N32K64', 'Fp8', 'Fp8')
    def h_v_mfma_f32_32x32x64_f8f6f4(self, t, m, op): return self.mfma(t, m, False, 'M32N32K64', 'Fp8', 'Fp8')

    def h_v_mfma_f32_32x32x8_bf16(self, t, m, op):
        a, b = self.src(t[1]), self.src(t[2])
        if t[3] == '0':
            # the literal-0 accumulator: a FragZero operand lowers to the `, 0` form (no registers)
            z = self.nv
            self.nv += 1
            self.lines.append(f'%v{z}: Frag16 = FragZero {{ ty: Frag16 }}')
            c = f'Val({z})'
        else:
            c = self.src(t[3])
        self.dst(t[0], 'Frag16', f'MfmaBf16Wide {{ a: {a}, b: {b}, c: {c} }}')
        return True

    def h_v_smfmac_f32_16x16x128_fp8_fp8(self, t, m, op):
        a, b, idx = self.src(t[1]), self.src(t[2]), self.src(t[3])
        c = self.src(t[0])
        self.dst(t[0], 'Frag4', f'SmfmacFp8_16x16x128 {{ a: {a}, b: {b}, idx: {idx}, c: {c} }}')
        return True

    # constants / zero pairs / selects / compares into SGPR pairs
    def h_v_mov_b64(self, t, m, op):
        if len(t) == 2 and t[1] == '0':
            self.dst(t[0], 'Frag2', 'FragZero { ty: Frag2 }')
            return True
        return False

    def h_v_cndmask_b32(self, t, m, op):
        # d = mask ? src1 : src0
        mask = t[3]
        r = reg(mask) if mask != 'vcc' else ('vcc', 0, 1, False)
        if r is None:
            return False
        key = (r[0], r[1])
        if key not in self.state:
            return False
        c = self.state[key][0]
        self.dst(t[0], 'I32', f'Select {{ ty: I32, cond: Val({c}), a: {self.src(t[2], "I32s")}, b: {self.src(t[1], "I32s")} }}')
        return True

    def cmp64(self, t, pred, ty):
        want = 'F32' if ty == 'F32' else 'I32'
        a, b = self.src(t[1], want), self.src(t[2], want)
        if t[0] == 'vcc':
            self.new('vcc', 0, 1, 'I1', f'Cmp {{ pred: {pred}, ty: {ty}, a: {a}, b: {b} }}')
            return True
        r = reg(t[0])
        if not r:
            return False
        self.new('s', r[1], r[2], 'I1', f'Cmp {{ pred: {pred}, ty: {ty}, a: {a}, b: {b} }}')
        return True

    def h_v_cmp_eq_u32(self, t, m, op): return self.cmp64(t, 'Eq', 'U32')
    def h_v_cmp_lt_u32(self, t, m, op): return self.cmp64(t, 'Lt', 'U32')
    def h_v_cmp_ge_f32(self, t, m, op): return self.cmp64(t, 'Ge', 'F32')

    def h_s_cmp_le_u32(self, t, m, op):
        self.new('scc', 0, 1, 'I1', f'Cmp {{ pred: Le, ty: U32, a: {self.src(t[0])}, b: {self.src(t[1])} }}')
        return True

    # kernel arguments (s_load from the kernarg segment s[0:1])
    def h_s_load_dword(self, t, m, op):
        if t[1] != 's[0:1]':
            # a scalar load through a pointer the code holds in an SGPR pair
            r = reg(t[1])
            base = self.read('s', r[1], 2)
            self.dst(t[0], 'I32', f'LoadScalar {{ base: {base}, byte_off: {imm(t[2])} }}')
            return True
        self.dst(t[0], 'I32', f'KernargI32 {{ offset: {imm(t[2])} }}')
        return True

    def h_s_load_dwordx2(self, t, m, op):
        if t[1] != 's[0:1]':
            return False
        self.dst(t[0], 'Ptr', f'KernargPtr {{ offset: {imm(t[2])} }}')
        return True

    # buffer / global memory
    def h_buffer_load_dwordx4_reg(self, t, m, dw):
        voff, rsrc, soff = self.src(t[1], 'I32'), self.rsrc(t[2]), self.src(t[3], 'I32')
        off = 0
        for x in m:
            if x.startswith('offset:'):
                off = int(x.split(':')[1], 0)
        self.dst(t[0], {1: 'F32', 4: 'Frag4'}[dw], f'BufferLoad {{ rsrc: {rsrc}, voff: Some({voff}), soff: {soff}, imm: {off}, dwords: {dw} }}')
        return True

    def h_buffer_store_dwordx4(self, t, m, op): return self.store(t, m, 4)
    def h_buffer_store_dword(self, t, m, op): return self.store(t, m, 1)

    def store(self, t, m, dw):
        val, voff, rsrc, soff = self.src(t[0], 'F32' if dw == 1 else None), self.src(t[1], 'I32'), self.rsrc(t[2]), self.src(t[3], 'I32')
        off = 0
        for x in m:
            if x.startswith('offset:'):
                off = int(x.split(':')[1], 0)
        self.stmt(f'BufferStore {{ rsrc: {rsrc}, voff: Some({voff}), soff: {soff}, imm: {off}, dwords: {dw}, value: {val} }}')
        return True

    def h_global_load_ubyte(self, t, m, op):
        voff, base = self.src(t[1], 'I32'), self.src(t[2])
        if self.vals[int(base[4:-1])][3] != 'Ptr':
            raise Unlift('global base is not a pointer value')
        off = 0
        for x in m:
            if x.startswith('offset:'):
                off = int(x.split(':')[1], 0)
        p = self.nv
        self.nv += 1
        self.lines.append(f'%v{p}: Ptr = PtrAdd {{ base: {base}, byte_off: {voff} }}')
        self.dst(t[0], 'I32', f'LoadGlobalN {{ base: Val({p}), byte_off: {off}, ty: U8, cache: CachePolicy {{ sc0: false, sc1: false, nt: false }} }}')
        return True
    # SALU (uniform values; pinned to SGPRs by the exact mode)
    def h_s_add_u32(self, t, m, op):
        if t[0] == 'm0':
            # the author's M0 write: s_add_u32 m0, a, b = SetM0 of an unpinned add (no instruction for the add)
            a, b = self.src(t[1], 'I32'), self.src(t[2], 'I32')
            v = self.nv
            self.nv += 1
            self.lines.append(f'%v{v}: I32 = AddI32({a}, {b})')
            self.stmt(f'SetM0 {{ value: Val({v}) }}')
            self.m0val = f'Val({v})'
            self.notes['m0 write -> SetM0'] += 1
            return True
        return self.bin(t, 'AddI32({a}, {b})', 'I32', 'I32')

    def h_s_sub_u32(self, t, m, op): return self.bin(t, 'Bin(SubU32, {a}, {b})', 'I32', 'I32')
    def h_s_mul_i32(self, t, m, op): return self.bin(t, 'MulI32({a}, {b})', 'I32', 'I32')
    def h_s_lshl_b32(self, t, m, op): return self.bin(t, 'ShlI32({a}, {b})', 'I32', 'I32')
    def h_s_lshr_b32(self, t, m, op): return self.bin(t, 'LshrI32({a}, {b})', 'I32', 'I32')
    def h_s_and_b32(self, t, m, op): return self.bin(t, 'AndI32({a}, {b})', 'I32', 'I32')
    def h_s_or_b32(self, t, m, op): return self.bin(t, 'Bin(OrB32, {a}, {b})', 'I32', 'I32')

    def h_s_mov_b32(self, t, m, op):
        if t[0] == 'm0':
            x = self.src(t[1], 'I32')
            self.stmt(f'SetM0 {{ value: {x} }}')
            self.m0val = x
            self.notes['m0 write -> SetM0'] += 1
            return True
        i = imm(t[1])
        if i is not None:
            self.dst(t[0], 'I32', f'MovS {{ ty: I32, a: I32({i if i < 2**31 else i - 2**32}) }}')
            return True
        try:
            x = float(t[1])
            self.dst(t[0], 'F32', f'MovS {{ ty: F32, a: F32({x!r}) }}')
            return True
        except ValueError:
            pass
        s = self.src(t[1], 'I32')
        self.dst(t[0], 'I32', f'MovS {{ ty: I32, a: {s} }}')
        return True

    def h_s_addk_i32(self, t, m, op):
        # the author's two-address SALU add of a 16-bit constant (s_addk_i32)
        s = self.src(t[0], 'I32')
        k = imm(t[1])
        k = k - 0x10000 if k is not None and k >= 0x8000 else k
        self.dst(t[0], 'I32', f'AddK {{ a: {s}, k: {k} }}')
        return True

    def h_s_cmp_ge_u32(self, t, m, op):
        self.new('scc', 0, 1, 'I1', f'Cmp {{ pred: Ge, ty: U32, a: {self.src(t[0])}, b: {self.src(t[1])} }}')
        return True

    def h_s_cmp_lt_i32(self, t, m, op):
        self.new('scc', 0, 1, 'I1', f'Cmp {{ pred: Lt, ty: I32, a: {self.src(t[0])}, b: {self.src(t[1])} }}')
        return True

    def h_s_cmp_eq_u32(self, t, m, op):
        self.new('scc', 0, 1, 'I1', f'Cmp {{ pred: Eq, ty: U32, a: {self.src(t[0])}, b: {self.src(t[1])} }}')
        return True

    def h_s_cselect_b32(self, t, m, op):
        if ('scc', 0) not in self.state:
            return False
        c = self.state[('scc', 0)][0]
        self.dst(t[0], 'I32', f'Select {{ ty: I32, cond: Val({c}), a: {self.src(t[1], "I32s")}, b: {self.src(t[2], "I32s")} }}')
        return True

    # memory to LDS (M0 = the LDS destination; `set_m0` in exact mode)
    def h_buffer_load_dwordx4(self, t, m, op):
        if 'lds' not in m:
            return self.h_buffer_load_dwordx4_reg(t, m, 4)
        return self.lds_dma(t, m, 4)

    def h_buffer_load_dword(self, t, m, op):
        if 'lds' not in m:
            return False
        return self.lds_dma(t, m, 1)

    def lds_dma(self, t, m, dw):
        voff = self.src(t[0], 'I32')
        rsrc = self.rsrc(t[1])
        soff = self.src(t[2], 'I32')
        off = 0
        for x in m:
            if x.startswith('offset:'):
                off = int(x.split(':')[1], 0)
        # exact mode: the DMA uses the current M0 (SetM0); its LDS address operand (for the verifier) is the value last set
        lds = getattr(self, 'm0val', 'I32(0)')
        self.stmt(f'BufferLoadLds {{ rsrc: {rsrc}, voff: Some({voff}), soff: {soff}, lds_addr: {lds}, imm: {off}, dwords: {dw} }}')
        return True

    # EXEC save / all-lanes / restore around the LDS-DMA issue (s_mov_b64 forms)
    def h_s_mov_b64(self, t, m, op):
        if t[1] == 'exec':
            r = reg(t[0])
            self.stmt(f'ExecSave {{ sgpr: {r[1]} }}')
            # the pair holds the saved mask, not a value of the program
            for i in range(2):
                self.state.pop(('s', r[1] + i), None)
            return True
        if t[0] == 'exec' and t[1] in ('-1', '0xffffffffffffffff'):
            self.stmt('ExecAll')
            return True
        if t[0] == 'exec' and imm(t[1]) is not None:
            # a 32-bit literal is ZERO-extended for a 64-bit SALU move (0xffffffff = lanes 0..31)
            self.stmt(f'ExecSet {{ mask: {imm(t[1]) & 0xffffffff} }}')
            return True
        if t[0] == 'exec':
            r = reg(t[1])
            self.stmt(f'ExecRestore {{ sgpr: {r[1]} }}')
            return True
        return False

    # sync / waits / priority
    def h_s_barrier(self, t, m, op):
        self.stmt('Barrier')
        return True

    def h_s_setprio(self, t, m, op):
        self.stmt(f'SetPrio {{ prio: {int(t[0], 0)} }}')
        return True

    def h_s_waitcnt(self, t, m, op):
        txt = ' '.join(t + m)
        vm = re.search(r'vmcnt\((\d+)\)', txt)
        lg = re.search(r'lgkmcnt\((\d+)\)', txt)
        if vm and lg:
            # the combined form: one s_waitcnt
            self.stmt(f'WaitCnt {{ vm: {vm.group(1)}, lgkm: {lg.group(1)} }}')
        elif vm:
            self.stmt(f'WaitVm {{ count: {vm.group(1)} }}')
        elif lg:
            self.stmt(f'WaitLds {{ count: {lg.group(1)} }}')
        self.notes['waitcnt (waitcnt(vm, lgkm) in exact mode)'] += 1
        return True

    def h_s_nop(self, t, m, op):
        self.stmt(f'SNop {{ count: {int(t[0], 0)} }}')
        self.notes['s_nop (exact mode: s_nop(n))'] += 1
        return True

    def pair_ok(self, a0, a1):
        # the pointer-add pair can be lifted only when its source pair reads as one pointer
        t0, _ = toks(a0)
        r0 = reg(t0[0])
        if not r0:
            return False
        try:
            mark, nv0, np0, st0, vw0 = len(self.lines), self.nv, len(self.pins), dict(self.state), dict(self.views)
            x = self.read('s', r0[1], 2)
            ok = x.startswith('Var(') or self.vals[int(x[4:-1])][3] == 'Ptr'
        except Unlift:
            ok = False
        if not ok:
            del self.lines[mark:]
            del self.pins[np0:]
            self.state, self.views = st0, vw0
        return ok

    def mask_fix(self, op, args):
        if not getattr(self, 'fix_mask', False):
            return
        t, m = toks(args)
        prev = getattr(self, 'prev_mask_write', None)
        self.prev_mask_write = None
        if prev and op.startswith('v_cndmask'):
            mask = t[3] if len(t) > 3 else 'vcc'
            if mask == prev:
                self.stmt('SNop { count: 0 }')
                self.notes['added s_nop 0 (lane-mask read right after its VALU write)'] += 1
                self.added_nops.append(f'{op} {args}')
        if op.startswith('v_cmp'):
            self.prev_mask_write = t[0]

    def lgkm_fix(self, op, args):
        # in-order LDS returns: a read of a register whose ds_read is still counted needs lgkmcnt(newer entries)
        if not getattr(self, 'fix_waits', False):
            return
        q = self.lgkm
        if op == 's_waitcnt':
            mm = re.search(r'lgkmcnt\((\d+)\)', args)
            if mm:
                n = int(mm.group(1))
                while len(q) > n:
                    q.pop(0)
            return
        d, u = defs_uses(op, args)
        if q and u:
            hit = [i for i, (_, regs) in enumerate(q) if regs & u]
            if hit:
                need = len(q) - 1 - max(hit)
                self.stmt(f'WaitLds {{ count: {need} }}')
                self.notes['added wait (timing-reliant LDS read)'] += 1
                self.added_waits.append(f'lgkmcnt({need}) before {op} {args}')
                while len(q) > need:
                    q.pop(0)
        if op.startswith('ds_read') or op.startswith('s_load'):
            q.append((op, d))

    def run(self, ins):
        i = 0
        while i < len(ins):
            sec, op, args = ins[i]
            # v_cvt_scalef32_pk_fp8_bf16 lo + hi into the same register -> one CvtP8x4FromU16 (its fixed 2-instruction expansion)
            if op == 'v_cvt_scalef32_pk_fp8_bf16' and i + 1 < len(ins) and ins[i + 1][1] == op:
                t0, m0 = toks(args)
                t1, m1 = toks(ins[i + 1][2])
                if t0[0] == t1[0] and not m0 and m1 == ['op_sel:[0,0,1]'] and t0[2] == t1[2]:
                    u0, u1, sc = self.src(t0[1]), self.src(t1[1]), self.src(t0[2])
                    self.dst(t0[0], 'U32', f'CvtP8x4FromU16({u0}, {u1}, {sc})')
                    self.cov[op] += 2
                    i += 2
                    continue
            if op == 's_add_u32' and i + 1 < len(ins) and ins[i + 1][1] == 's_addc_u32' and self.pair_ok(args, ins[i + 1][2]):
                t0, _ = toks(args)
                t1, _ = toks(ins[i + 1][2])
                r0, r1 = reg(t0[0]), reg(t1[0])
                if r0 and r1 and r1[1] == r0[1] + 1 and t0[1] == t0[0] and t1[1] == t1[0] and t1[2] == '0':
                    base = self.read('s', r0[1], 2)
                    self.new('s', r0[1], 2, 'Ptr', f'PtrAdd {{ base: {base}, byte_off: {self.src(t0[2])} }}')
                    self.cov['s_add_u32'] += 1
                    self.cov['s_addc_u32'] += 1
                    i += 2
                    continue
            if op.startswith('s_cbranch') or op == 's_branch' or op == 's_endpgm':
                self.stmt(f'# {op} {args} (structured control flow: the shell)')
                self.notes['branch (structured shell)'] += 1
                i += 1
                continue
            self.lgkm_fix(op, args)
            self.mask_fix(op, args)
            # M0 written right before the LDS-DMA that reads it: one wait state (named deviation, --fix-m0)
            if getattr(self, 'fix_m0', False) and 'lds' in args and op.startswith('buffer_load') and getattr(self, 'prev_m0', False):
                self.stmt('SNop { count: 0 }')
                self.notes['added s_nop 0 (M0 write right before its LDS-DMA)'] += 1
                self.added_nops.append(f'{op} {args}')
            self.prev_m0 = args.startswith('m0,') and op in ('s_add_u32', 's_mov_b32')
            # the author's VOP3 encoding of a VALU instruction (also where VOP2 would encode)
            self.cur_e64 = op.startswith('v_') and op.endswith('_e64')
            self.lift(op, args)
            self.cur_e64 = False
            i += 1

# ---- control flow: basic blocks and the structured form (exact mode E5: exit_if, cold_at, begin_if_br; scope variables) ----

def cfg(ins):
    """Basic blocks of the instruction list (sec, op, args): (start, end, label), successors (taken first, then fall-through)."""
    labels = {}
    for k, (sec, op, args) in enumerate(ins):
        labels.setdefault(sec, k)
    starts = {0} | set(labels.values())
    def target(args):
        return labels.get(args.strip())
    for k, (sec, op, args) in enumerate(ins):
        if op.startswith('s_cbranch') or op in ('s_branch', 's_endpgm'):
            starts.add(k + 1)
            if target(args) is not None:
                starts.add(target(args))
    starts = sorted(s for s in starts if s < len(ins))
    name = {v: k for k, v in labels.items()}
    bl = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(ins)
        bl.append((s, e, name.get(s, f'@{s}')))
    idx = {s: i for i, (s, _, _) in enumerate(bl)}
    succ = collections.defaultdict(list)
    endblk = next((i for i, (s, e, _) in enumerate(bl) if any(ins[k][1] == 's_endpgm' for k in range(s, e))), len(bl) - 1)
    for i, (s, e, _) in enumerate(bl):
        op, args = ins[e - 1][1], ins[e - 1][2]
        if (op.startswith('s_cbranch') or op == 's_branch') and target(args) is not None:
            succ[i].append(idx[target(args)])
        elif op.startswith('s_cbranch') or op == 's_branch':
            # a branch (conditional too) to a label without a symbol: the kernel's end (the block with s_endpgm); checked by
            # tools/isa_branch_targets.py against the reference
            succ[i].append(endblk)
        if op not in ('s_branch', 's_endpgm') and i + 1 < len(bl):
            succ[i].append(i + 1)
    return bl, succ


def dominators(n, succ, entry):
    pred = collections.defaultdict(list)
    for a in range(n):
        for b in succ.get(a, []):
            pred[b].append(a)
    dom = {i: set(range(n)) for i in range(n)}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i == entry:
                continue
            ps = [dom[p] for p in pred[i]]
            d = (set.intersection(*ps) if ps else set()) | {i}
            if d != dom[i]:
                dom[i] = d
                changed = True
    return dom


def structure(bl, succ):
    """Structured tree of a reducible CFG (hand-scheduled kernel shapes):
    ('block', i) | ('if', cond_block, then_nodes, else_nodes) | ('loop', header, body_nodes) | ('exit_if', target, nodes) |
    ('cold_if', block, nodes) | ('continue',) | ('end',). Loop back edges by dominance; branches into blocks that jump back into the
    loop are cold ifs; other forward branches are ifs joined at the immediate post-dominator."""
    n = len(bl)
    dom = dominators(n, succ, 0)
    # post-dominators: reverse graph with a virtual exit n
    rsucc = collections.defaultdict(list)
    for a in range(n):
        ss = succ.get(a, [])
        if not ss:
            rsucc[n].append(a)
        for b in ss:
            rsucc[b].append(a)
    pdom = dominators(n + 1, rsucc, n)
    def ipdom(i):
        c = pdom[i] - {i}
        # the closest: the post-dominator post-dominated by all the others
        for x in c:
            if all(y in pdom[x] for y in c):
                return x if x < n else None
        return None
    loops = {}
    for a in range(n):
        for h in succ.get(a, []):
            if h in dom[a]:
                pred = collections.defaultdict(list)
                for x in range(n):
                    for y in succ.get(x, []):
                        pred[y].append(x)
                body = {h}
                st = [a]
                while st:
                    x = st.pop()
                    if x not in body:
                        body.add(x)
                        st.extend(pred[x])
                loops[h] = body
    def loop_of(i):
        best = None
        for h, body in loops.items():
            if i in body and (best is None or len(body) < len(loops[best])):
                best = h
        return best
    used = set()

    def seq(i, stop, loop):
        nodes = []
        while i is not None and i != stop:
            if i in loops and loop != i and (loop is None or i != loop):
                exits = {y for x in loops[i] for y in succ.get(x, []) if y not in loops[i]}
                c = set.intersection(*[pdom[x] for x in exits]) if exits else set()
                j = next((x for x in c if all(y in pdom[x] for y in c)), None)
                j = j if j is not None and j < n else None
                body_nodes = seq_body(i, j)
                nodes.append(('loop', i, body_nodes))
                i = j                   # the exits' bodies run to the loop's post-dominator, where the code continues
                continue
            if i in used:
                nodes.append(('jump', i))
                return nodes
            used.add(i)
            nodes.append(('block', i))
            ss = succ.get(i, [])
            if loop is not None and loop in ss:
                nodes.append(('continue',))
                return nodes
            if len(ss) == 2:
                taken, fall = ss
                if loop is not None and taken not in loops[loop]:
                    nodes.append(('exit_if', taken, seq(taken, None, None)))
                    i = fall
                    continue
                if loop is not None and taken in loops[loop] and taken > max(x for x in loops[loop] if loop in succ.get(x, [])):
                    # out-of-line block of this loop: runs, then jumps back to the block after the branch's fall-through run
                    used.add(taken)
                    nodes.append(('cold_if', taken, [('block', taken)]))
                    i = fall
                    continue
                j = ipdom(i)
                nodes.append(('if', i, seq(fall, j, loop), seq(taken, j, loop)))
                i = j
                continue
            if len(ss) == 1:
                i = ss[0]
                continue
            nodes.append(('end',))
            return nodes
        return nodes

    def seq_body(h, follow):
        nodes = []
        i = h
        first = True
        while True:
            if i in used:
                nodes.append(('jump', i))
                return nodes
            used.add(i)
            nodes.append(('block', i))
            ss = succ.get(i, [])
            if h in ss:
                nodes.append(('continue',))
                return nodes
            if len(ss) == 2:
                taken, fall = ss
                if taken not in loops[h]:
                    nodes.append(('exit_if', taken, seq(taken, follow, None)))
                    i = fall
                    continue
                if taken > i and len(succ.get(taken, [])) == 1 and succ[taken][0] < taken:
                    # out-of-line block: it REPLACES the hot blocks from the fall-through up to its jump-back target (an if / else whose
                    # then-branch is placed out of line); both arms join there
                    used.add(taken)
                    join = succ[taken][0]
                    els, x = [], fall
                    while x != join:
                        els.append(('block', x))
                        used.add(x)
                        x = succ[x][-1]
                    nodes.append(('cold_if', taken, [('block', taken)], els))
                    i = join
                    continue
                j = ipdom(i)
                nodes.append(('if', i, seq(fall, j, h), seq(taken, j, h)))
                i = j
                continue
            i = ss[0]
    tree = seq(0, None, None)
    return tree, loops, {x for h, b in loops.items() for x in b if any(x > a for a in b if h in succ.get(a, []))}

def print_tree(nodes, bl, ind=0):
    for nd in nodes:
        k = nd[0]
        if k == 'block':
            s, e, n = bl[nd[1]]
            print('  ' * ind + f'block {n} ({e - s})')
        elif k == 'loop':
            print('  ' * ind + f'loop {bl[nd[1]][2]}')
            print_tree(nd[2], bl, ind + 1)
        elif k == 'cold_if':
            print('  ' * ind + f'cold_if -> {bl[nd[1]][2]} (out of line, jumps back) else {{')
            print_tree(nd[3] if len(nd) > 3 else [], bl, ind + 1)
            print('  ' * ind + '}')
        elif k == 'exit_if':
            print('  ' * ind + f'exit_if -> {bl[nd[1]][2]} {{')
            print_tree(nd[2], bl, ind + 1)
            print('  ' * ind + '}')
        elif k == 'if':
            print('  ' * ind + f'if_br (branch of {bl[nd[1]][2]}) {{')
            print_tree(nd[2], bl, ind + 1)
            print('  ' * ind + '} else {')
            print_tree(nd[3], bl, ind + 1)
            print('  ' * ind + '}')
        elif k in ('continue', 'end'):
            print('  ' * ind + k)
        elif k == 'jump':
            print('  ' * ind + f'(joins {bl[nd[1]][2]})')

# ---- whole kernel: register liveness across blocks, scope variables, the structured shell (E5) ----

def defs_uses(op, args):
    """(defs, uses) as sets of (file, register) for v / s registers (exec / vcc / scc / m0 are handled by the forms)."""
    t, m = toks(args)
    if op.startswith(('s_cbranch', 's_branch', 's_endpgm', 's_waitcnt', 's_nop', 's_barrier', 's_setprio')):
        return set(), set()
    lds_dma = op.startswith('buffer_load') and 'lds' in m
    no_dst = op.startswith(NO_DST_OPS) or lds_dma or op.startswith('v_cmp') or op.startswith('s_cmp')
    def rr(tok):
        r = reg(tok)
        if not r:
            return []
        f, a, n, _ = r
        return [(f, a + i) for i in range(n)]
    d, u = set(), set()
    if t and not no_dst:
        d = set(rr(t[0]))
        srcs = t[1:]
    else:
        srcs = t
        if op.startswith('v_cmp') and t and t[0] not in ('vcc',):
            d = set(rr(t[0]))
            srcs = t[1:]
    for x in srcs:
        u |= set(rr(x))
    # read-modify-write forms
    if op.startswith(('s_addk', 'v_smfmac')) or any(x.startswith('op_sel:[0,0,1]') for x in m) or op.startswith('v_permlane'):
        u |= d
    if op.startswith('v_permlane'):
        d |= set(rr(t[1]))
    return d, u


NO_DST_OPS = ('buffer_store', 'global_store', 'ds_write', 'ds_store')


def liveness(ins, bl, succ):
    n = len(bl)
    use, dfn = [set() for _ in range(n)], [set() for _ in range(n)]
    for b, (s, e, _) in enumerate(bl):
        for k in range(s, e):
            d, u = defs_uses(ins[k][1], ins[k][2])
            use[b] |= (u - dfn[b])
            dfn[b] |= d
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for b in reversed(range(n)):
            out = set()
            for x in succ.get(b, []):
                out |= live_in[x]
            inn = use[b] | (out - dfn[b])
            if out != live_out[b] or inn != live_in[b]:
                live_out[b], live_in[b] = out, inn
                changed = True
    return live_in, live_out


def cond_for(L, bl_ins, polarity_taken):
    """The I1 value for a block's conditional branch: True exactly when the branch is taken (polarity_taken) or not taken."""
    op = bl_ins[-1][1]
    key = ('scc', 0) if 'scc' in op else ('vcc', 0)
    if key not in L.state:
        raise Unlift(f'{op}: no condition value')
    c = f'Val({L.state[key][0]})'
    if key == ('vcc', 0):
        # s_cbranch_vcc(n)z: the wave branches on 'any lane' of the vector compare (exact mode: the VCC-pinned I1 is branched on directly)
        a = L.nv
        L.nv += 1
        L.lines.append(f'%v{a}: I1 = AnyLane({c})')
        c = f'Val({a})'
    taken_when_set = op.endswith(('scc1', 'vccnz'))
    want_set = taken_when_set == polarity_taken
    if want_set:
        return c
    w = L.nv
    L.nv += 1
    L.lines.append(f'%v{w}: I1 = NotI1({c})')
    return f'Val({w})'


def emit_kernel(path, exact, plan=None):
    """Whole-kernel structured lift. Registers live across a block boundary are scope variables: one I32 variable per dword, except
    SGPR pairs / quads that hold ONE pointer / descriptor value at every end of the blocks reaching a boundary - those are carried as one
    Ptr / Rsrc variable (provenance visible to gpu_verify). `plan` = the pointer / descriptor tuples seen at each block end (first pass)."""
    ins = parse(path)
    bl, succ = cfg(ins)
    tree, loops, cold = structure(bl, succ)
    live_in, live_out = liveness(ins, bl, succ)
    nb = len(bl)
    pred = collections.defaultdict(list)
    for a in range(nb):
        for b in succ.get(a, []):
            pred[b].append(a)
    g_end = plan or [set() for _ in range(nb)]
    # tuples carried INTO each block as one variable: seen at the end of every predecessor, every dword live-in
    g_in = []
    for b in range(nb):
        if not pred[b]:
            g_in.append(set())
            continue
        common = set.intersection(*[set(g_end[x]) for x in pred[b]])
        g_in.append({g for g in common if all(('s', g[0] + i) in live_in[b] for i in range(g[2]))})
    # a tuple a block hands on as one variable: some successor takes it so
    g_out = [set().union(*[g_in[x] for x in succ.get(b, [])]) if succ.get(b) else set() for b in range(nb)]
    groups = sorted(set().union(*g_in)) if g_in else []
    # registers written exactly once, by an instruction of a TOP-LEVEL block (the prelude): carried by their value (visible to the
    # whole function; gpu_verify sees the value itself, not a variable)
    def_count = collections.Counter()
    for (_, op, args) in ins:
        for d in defs_uses(op, args)[0]:
            def_count[d] += 1
    top_blocks = {nd[1] for nd in tree if nd[0] == 'block'}
    single = set()
    if '--values-for-single-defs' in sys.argv:
        # every definition in a top-level block: after the prelude the register holds one fixed value (its last top-level def)
        top_defs = collections.Counter()
        for b0 in top_blocks:
            s0, e0, _ = bl[b0]
            for k in range(s0, e0):
                for d in defs_uses(ins[k][1], ins[k][2])[0]:
                    top_defs[d] += 1
        single = {d for d, n in top_defs.items() if n == def_count[d]}
    globals_ = {}
    wg_regs = [('s', int(x[1:])) for x in sys.argv[sys.argv.index('--wgmap') + 1].split(',')] if '--wgmap' in sys.argv else []
    wg_vals = []
    L = Lifter()
    L.single = single
    L.exact = exact
    L.fix_waits = '--fix-waits' in sys.argv
    L.fix_mask = '--fix-lanemask' in sys.argv
    L.fix_m0 = '--fix-m0' in sys.argv
    carried = sorted(set().union(*[live_in[b] for b in range(nb)]) - single)
    var_tys = []
    var_slots = [(f, r, 1, 'I32') for (f, r) in carried] + [('s', a, n, ty) for (a, ty, n) in groups]
    undefs = {}
    for ty in sorted({x[3] for x in var_slots}):
        u = L.nv
        L.nv += 1
        L.shell.append(f'%v{u}: {ty} = Undef {{ ty: {ty} }}')
        undefs[ty] = u
    pv_of = {}
    for k, (f, r, n, ty) in enumerate(var_slots):
        pv = L.nv
        L.nv += 1
        L.vals[pv] = (f, r, n, ty)
        L.varof[pv] = k
        L.varreg[k] = (f, r)
        pv_of[k] = pv
        var_tys.append(ty)
        L.shell_pins.append(f'reg_plan {"PinV" if f == "v" else "PinS"} {{ r: Var(VarId({k})), base: {r} }}')
    if var_slots:
        L.shell.append('DeclareVars { vars: [' + ', '.join(f'VarId({k})' for k in range(len(var_slots))) + '], init: [' + ', '.join(f'Val({undefs[x[3]]})' for x in var_slots) + '] }')
    dword_k = {(f, r): k for k, (f, r, n, ty) in enumerate(var_slots) if n == 1}
    group_k = {(f_r[1], f_r[3], f_r[2]): k for k, f_r in enumerate(var_slots) if f_r[2] > 1}
    entry_hw = sorted(live_in[0])
    anchors = []
    placed = set()
    seen_end = [set() for _ in range(nb)]

    def block(b):
        s, e, _ = bl[b]
        L.state = {}
        for (f, r) in live_in[b]:
            if (f, r) in globals_:
                L.state[(f, r)] = globals_[(f, r)]
            elif (f, r) in dword_k:
                L.state[(f, r)] = (pv_of[dword_k[(f, r)]], 0)
        # --wgmap: the workgroup-map registers, rebound at the first top-level block after their (branchy) definitions to zero-cost
        # views, so the contract's WgMap can name them as values
        if wg_regs and b in top_blocks and b != 0 and not wg_vals and all(r in live_in[b] for r in wg_regs):
            for r in wg_regs:
                k = dword_k[r]
                v = L.new(r[0], r[1], 1, 'I32', f'BitcastI32(Var(VarId({k})))')
                globals_[r] = (v, 0)
                wg_vals.append(v)
        for g in g_in[b]:
            k = group_k[g]
            for i in range(g[2]):
                L.state[('s', g[0] + i)] = (pv_of[k], i)
        L.views = {}
        L.lgkm = []
        if b == 0:
            # hardware-initialised registers: v0 = workitem id, s[0:1] = kernarg pointer, then the enabled workgroup ids
            L.new('v', 0, 1, 'U32', 'WorkitemId')
            L.new('s', 0, 2, 'Ptr', 'KernargSegment')
            for i, wg in enumerate(['WorkgroupIdX', 'WorkgroupIdY', 'WorkgroupIdZ']):
                if ('s', 2 + i) in live_in[0]:
                    L.new('s', 2 + i, 1, 'U32', wg)
        body = ins[s:e]
        last = body[-1][1]
        run = body[:-1] if last.startswith(('s_cbranch', 's_branch')) else body
        L.run([x for x in run if x[1] != 's_endpgm'])
        # single-def registers defined here: their value is the register's content for the rest of the kernel
        for d in single:
            if d in L.state and L.state[d][0] not in L.varof:
                globals_[d] = L.state[d]
        # pointer / descriptor tuples at the block end (the plan of the second pass)
        for (f, r), (v, o) in L.state.items():
            if f != 's' or o != 0 or v not in L.vals:
                continue
            _, base, n, ty = L.vals[v]
            if ty in ('Ptr', 'Rsrc') and base == r and all(L.state.get(('s', r + i)) == (v, i) for i in range(n)) and all(('s', r + i) in live_out[b] for i in range(n)):
                seen_end[b].add((r, ty, n))
        # hand on: tuple variables first, then the remaining dwords
        covered = set()
        for g in sorted(g_out[b]):
            k = group_k[g]
            covered |= {('s', g[0] + i) for i in range(g[2])}
            if all(L.state.get(('s', g[0] + i)) == (pv_of[k], i) for i in range(g[2])):
                continue
            try:
                x = L.read('s', g[0], g[2])
            except Unlift as ex:
                L.notes[f'carried tuple not assembled ({ex})'] += 1
                continue
            L.lines.append(f'SetVar {{ var: VarId({k}), value: {x} }}')
        for (f, r) in sorted(live_out[b]):
            if (f, r) in covered or (f, r) not in dword_k:
                continue
            k = dword_k[(f, r)]
            cur = L.state.get((f, r))
            if cur is None or cur == (pv_of[k], 0):
                continue
            try:
                x = L.coerce(L.read(f, r, 1), 'I32')
            except Unlift as ex:
                L.notes[f'carried dword not readable ({ex})'] += 1
                continue
            L.lines.append(f'SetVar {{ var: VarId({k}), value: {x} }}')
        return body

    def emit(nodes, loopname=None):
        i = 0
        while i < len(nodes):
            nd = nodes[i]
            k = nd[0]
            if k == 'block':
                body = block(nd[1])
                nxt = nodes[i + 1] if i + 1 < len(nodes) else None
                if nxt and nxt[0] == 'if':
                    c = cond_for(L, body, False)
                    n_anchor = len(anchors)
                    L.lines.append(f'if {c} {{')
                    emit(nxt[2], loopname)
                    if nxt[3]:
                        L.lines.append('} else {')
                        emit(nxt[3], loopname)
                    L.lines.append('}')
                    # out-of-line cold blocks of the loops inside this if / else: laid out where the if's arms end (before its join),
                    # the hand-scheduled layout
                    if nxt[3]:
                        for a_ in anchors[n_anchor:]:
                            if a_.startswith('cold_if') and a_ not in placed:
                                L.lines.append(f'LayoutAnchor {{ name: "{a_}" }}')
                                placed.add(a_)
                    i += 2
                    continue
                if nxt and nxt[0] in ('exit_if', 'cold_if'):
                    c = cond_for(L, body, True)
                    anchor = f'{nxt[0]}_{loopname}'
                    if anchor not in anchors:
                        anchors.append(anchor)
                    L.lines.append(f'BranchPlace {{ anchor: "{anchor}" }}')
                    L.lines.append(f'if {c} {{')
                    emit(nxt[2], loopname)
                    if nxt[0] == 'exit_if':
                        L.lines.append(f'BreakIf {{ cond: {c} }}')
                    if nxt[0] == 'cold_if' and len(nxt) > 3 and nxt[3]:
                        L.lines.append('} else {')
                        emit(nxt[3], loopname)
                    L.lines.append('}')
                    i += 2
                    continue
            elif k == 'loop':
                name = bl[nd[1]][2]
                L.lines.append('loop vars [] init [] {')
                emit(nd[2], name)
                L.lines.append('}')
                if f'exit_if_{name}' in anchors:
                    L.lines.append(f'LayoutAnchor {{ name: "exit_if_{name}" }}')
            i += 1
    emit(tree)
    # out-of-line cold blocks: at the end of the kernel's hot code
    for a in anchors:
        if a.startswith('cold_if') and a not in placed:
            L.lines.append(f'LayoutAnchor {{ name: "{a}" }}')
    # iterate the plan to a fixpoint: a tuple handed on unchanged through a block (a loop latch that never touches a descriptor) is seen
    # at its end only once the tuple is carried INTO it, so the plan grows pass by pass
    if plan is None or [set(x) for x in seen_end] != [set(x) for x in plan]:
        rounds = getattr(emit_kernel, 'rounds', 0) + 1
        emit_kernel.rounds = rounds
        if rounds < 8:
            merged = [set(seen_end[i]) | (set(plan[i]) if plan else set()) for i in range(nb)]
            return emit_kernel(path, exact, merged)
    L.wg_vals = wg_vals
    return L, var_tys, entry_hw


def main():
    path = sys.argv[1]
    secs = sys.argv[sys.argv.index('--sections') + 1].split(',') if '--sections' in sys.argv else []
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'lifted'
    if '--structure' in sys.argv:
        allins = parse(path)
        bl, succ = cfg(allins)
        tree, loops, cold = structure(bl, succ)
        print(f'{len(bl)} blocks, {len(loops)} natural loops, {len(cold)} out-of-line blocks')
        print_tree(tree, bl)
        return
    var_tys = []
    if '--kernel' in sys.argv:
        L, var_tys, entry_hw = emit_kernel(path, '--exact' in sys.argv)
        ins = parse(path)
        print('entry live-in registers (hardware-initialised):', ', '.join(f'{f}{r}' for f, r in entry_hw))
        for w in L.added_waits:
            print('  added wait:', w)
        for w in L.added_nops:
            print('  added s_nop 0 before:', w)
    else:
        ins = [x for x in parse(path) if x[0] in secs]
        L = Lifter()
        L.run(ins)
    total = len(ins)
    lifted = sum(L.cov.values())
    if '--report' in sys.argv or '--out' not in sys.argv:
        print(f'instructions {total}; lifted to DSL statements {lifted} ({100.0 * lifted / max(total, 1):.1f} %); unmapped {sum(L.unmapped.values())}')
        for k, v in L.unmapped.most_common():
            print(f'  unmapped {v:4d} {k}')
        for k, v in L.notes.most_common():
            print(f'  note     {v:4d} {k}')
        print(f'  values {L.nv}, pins {len([p for p in L.pins if p.startswith("reg_plan")])}, shell (live-in) values {len(L.shell)}')
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
        with open(out, 'w') as f:
            f.write('irv1\n')
            f.write(f'dsl "{name}"\n')
            opt = lambda k, d: int(sys.argv[sys.argv.index(k) + 1], 0) if k in sys.argv else d
            # the input kernel's interface (from its code-object metadata, passed by the caller): kernarg size, workgroup size, LDS bytes
            f.write(f'kernarg_size {opt("--kernarg-size", 8)}  workgroup {opt("--workgroup", 512)}  lds {opt("--lds", 163840)}  value_count {L.nv}\n')
            f.write('profile None  lower_opts []  sched_opts None\n')
            if '--exact' in sys.argv:
                f.write('exact\n')
            f.write('var_tys [' + ', '.join(var_tys) + ']\n')
            # one argument per kernarg word the code loads (I32 / Ptr at its offset); with --contract-from, the named arguments and the
            # argument-level contract (extents, kernarg ranges / definitions, grid definitions) of an IR of the SAME kernarg layout (our
            # own replica's), so the verifier bounds the accesses through them
            body = '\n'.join(L.shell + L.lines)
            ka = {}
            for k, off in re.findall(r'Kernarg(I32|Ptr) \{ offset: (\d+) \}', body):
                ka.setdefault(int(off), k)
            if not ka:
                ka = {0: 'Ptr'}
            named, contract = {}, None
            if '--contract-from' in sys.argv:
                src = open(sys.argv[sys.argv.index('--contract-from') + 1]).read().split('\n')
                for l in src:
                    m = re.match(r'^arg Arg \{ name: "(\w+)", ty: (\w+), kernarg_offset: (\d+), size: (\d+) \}', l)
                    if m:
                        named[int(m.group(3))] = l
                i0 = next(i for i, l in enumerate(src) if l.startswith('contract Contract {'))
                keep = {}
                i = i0 + 1
                while src[i] != '}':
                    m = re.match(r'^    (\w+): (.*)$', src[i])
                    j = i
                    if m and m.group(2).endswith(('[', '{', '(')):
                        while not re.match(r'^    [\]\})],?$', src[j]):
                            j += 1
                    keep[m.group(1) if m else f'_{i}'] = src[i:j + 1]
                    i = j + 1
                use = ('extents', 'ka_ranges', 'grid_defs', 'ka_defs')
                contract = ['contract Contract {']
                for k, lines in keep.items():
                    if k in use:
                        contract += lines
                    elif k.startswith('_'):
                        continue
                    elif k == 'wg_maps' and getattr(L, 'wg_vals', None):
                        reason = sys.argv[sys.argv.index('--wgmap-reason') + 1] if '--wgmap-reason' in sys.argv else 'workgroup map'
                        outs = ', '.join(str(v) for v in L.wg_vals)
                        axes = ', '.join(str(i) for i in range(len(L.wg_vals)))
                        contract.append(f'    wg_maps: [ WgMap {{ outs: [{outs}], axes: [{axes}], reason: {reason!r}, union: false, range: None }} ],'.replace("'", '"'))
                    else:
                        # facts about the other kernel's values / tiles do not carry over
                        empty = {'tile_trust': 'None'}.get(k, '[]')
                        contract.append(f'    {k}: {empty},')
                contract.append('}')
            for off in sorted(set(ka) | set(named)):
                if off in named:
                    f.write(named[off] + '\n')
                else:
                    k = ka[off]
                    f.write(f'arg Arg {{ name: "ka_{off:x}", ty: {k}, kernarg_offset: {off}, size: {8 if k == "Ptr" else 4} }}\n')
            if contract:
                f.write('\n'.join(contract) + '\n')
            for p in L.shell_pins + L.pins:
                f.write(p + '\n')
            defined = set(int(x) for x in re.findall(r'^%v(\d+):', '\n'.join(L.shell + L.lines), re.M))
            gaps = [i for i in range(L.nv) if i not in defined]
            if gaps:
                f.write('values { ' + ' '.join(f'%v{i}: I32' for i in gaps) + ' }\n')
            f.write('body {\n')
            # the live-in shell's scalar loads complete before the lifted code (an authored wait)
            ind = 1
            for s in L.shell + (['WaitLds { count: 0 }'] if '--kernel' not in sys.argv else []) + L.lines:
                if s.startswith('}'):
                    ind -= 1
                f.write('  ' * ind + s + '\n')
                if s.endswith('{') and not s.startswith('%'):
                    ind += 1
            f.write('}\n')


if __name__ == '__main__':
    main()
