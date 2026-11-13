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
RTY = {'ConstI32': 'I32', 'ConstF32': 'F32', 'KernargI32': 'I32', 'KernargPtr': 'Ptr', 'BufferRsrc': 'Rsrc', 'BufferRsrcDyn': 'Rsrc', 'AddI32': 'I32', 'MulI32': 'I32', 'ShlI32': 'I32', 'LshrI32': 'I32',
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
        if op == 'Select':
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
        self.nv = 0
        self.vals = {}           # id -> (file, base, n, ty)
        self.state = {}          # (file, reg) -> (value id, offset)
        self.views = {}          # (id, start, n) -> view id
        self.cov = collections.Counter()
        self.unmapped = collections.Counter()
        self.notes = collections.Counter()

    def new(self, f, base, n, ty, rhs, shell=False):
        v = self.nv
        self.nv += 1
        ty = rty(rhs, ty)
        (self.shell if shell else self.lines).append(f'%v{v}: {ty} = {rhs}')
        self.vals[v] = (f, base, n, ty)
        if f == 'v':
            self.pins.append(f'reg_plan PinV {{ r: Val({v}), base: {base} }}')
        else:
            # SGPR pins: the exact mode's `pin_sgpr_at` (E3); recorded as a comment until its irv1 spelling exists
            self.pins.append(f'# pin_sgpr {v} s{base} n{n}')
        for i in range(n):
            self.state[(f, base + i)] = (v, i)
        return v

    def livein(self, f, base, n):
        if f == 's':
            if n == 4:
                p = self.nv
                self.nv += 1
                self.shell.append(f'%v{p}: Ptr = KernargPtr {{ offset: 0 }}')
                return self.new('s', base, n, 'Rsrc', f'BufferRsrc {{ base: Val({p}), num_records: 4294967295 }}', shell=True)
            return self.new('s', base, n, 'U32' if n == 1 else 'Ptr', 'KernargI32 { offset: 0 }' if n == 1 else 'KernargPtr { offset: 0 }', shell=True)
        if n == 8:
            parts = []
            for h in range(2):
                q = self.nv
                self.nv += 1
                self.shell.append(f'%v{q}: Frag4 = FragZero {{ ty: Frag4 }}')
                self.pins.append(f'reg_plan PinV {{ r: Val({q}), base: {base + 4 * h} }}')
                parts.append(f'Val({q})')
            return self.new('v', base, n, 'Frag8', f'FragCat {{ parts: [{parts[0]}, {parts[1]}], dwords: 8 }}', shell=True)
        rhs = 'ConstI32(0)' if n == 1 else f'FragZero {{ ty: {FRAG[n]} }}'
        return self.new('v', base, n, FRAG.get(n, 'U32'), rhs, shell=True)

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
        if o0 == 0 and vn == n and all(self.state[(f, base + i)] == (v0, i) for i in range(n)):
            return f'Val({v0})'
        if all(self.state[(f, base + i)] == (v0, o0 + i) for i in range(n)):
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
                self.pins.append(f'reg_plan PinV {{ r: Val({v}), base: {base} }}')
                self.views[key] = v
            return f'Val({self.views[key]})'
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
        self.pins.append(f'reg_plan PinV {{ r: Val({v}), base: {base} }}')
        return f'Val({v})'

    def coerce(self, x, want):
        # a 32-bit value used as another 32-bit type: a zero-instruction bitcast pinned to the same register
        m = re.match(r'Val\((\d+)\)$', x)
        if not m or want is None:
            return x
        v = int(m.group(1))
        f, base, n, ty = self.vals[v]
        if n != 1 or ty == want or (want == 'I32' and ty in ('I32',)) :
            return x
        if want == 'F32' and ty in ('U32', 'I32'):
            op = 'BitcastF32'
        elif want in ('I32', 'U32') and ty == 'F32':
            op = 'BitcastI32'
        elif want == 'I32' and ty == 'U32':
            op = 'BitcastI32'
        else:
            return x
        key = ('cast', v, want)
        if key not in self.views:
            w = self.nv
            self.nv += 1
            rt = 'F32' if op == 'BitcastF32' else 'I32'
            self.lines.append(f'%v{w}: {rt} = {op}(Val({v}))')
            self.vals[w] = (f, base, 1, rt)
            if f == 'v':
                self.pins.append(f'reg_plan PinV {{ r: Val({w}), base: {base} }}')
            self.views[key] = w
            self.notes['bitcast view'] += 1
        return f'Val({self.views[key]})'

    def rsrc(self, tok):
        # a buffer descriptor operand: an Rsrc value, or four SGPRs the code assembled itself -> BufferRsrcDyn over its base / size words
        f, a, n, _ = reg(tok)
        x = self.read(f, a, n)
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
            if neg:
                self.notes['neg-modifier'] += 1
            return self.coerce(x, want)
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
            if want in ('I32', 'U32'):
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
            self.dst(t[0], 'I32', f'ConstI32({i if i < 2**31 else i - 2**32})')
            return True
        try:
            x = float(t[1])
            self.dst(t[0], 'F32', f'ConstF32({x!r})')
            return True
        except ValueError:
            pass
        # a register copy: the DSL has no "mov" op (copies are the allocator's); count it, express as a view of the source value
        s = self.src(t[1], 'I32')
        f, a, n, _ = reg(t[0])
        self.notes['mov-as-op'] += 1
        self.new(f, a, n, 'U32', f'AddI32({s}, I32(0))')
        return False

    def h_v_pk_mul_f32(self, t, m, op): return self.bin(t, 'PkMulF32({a}, {b})', 'Frag2')

    def h_v_pk_fma_f32(self, t, m, op):
        a, b, c = self.src(t[1]), self.src(t[2]), self.src(t[3])
        self.dst(t[0], 'Frag2', f'PkFmaF32({a}, {b}, {c})')
        return True

    def h_v_permlane32_swap_b32(self, t, m, op):
        # swaps two registers in place: both are redefined
        a, b = self.src(t[0]), self.src(t[1])
        self.notes['permlane-swap'] += 1
        return False

    def h_v_cvt_scalef32_pk_fp8_bf16(self, t, m, op):
        # the lo / hi pair is one DSL op (CvtP8x4FromU16 = 2 instructions); the lifter pairs them in `run`
        return False

    def h_v_cmp_nge_f32(self, t, m, op):
        if t[0] != 'vcc':
            return False
        a, b = self.src(t[1], 'F32'), self.src(t[2], 'F32')
        c = self.new('vcc', 0, 1, 'I1', f'Cmp {{ pred: Ge, ty: F32, a: {a}, b: {b} }}')
        self.notes['cmp-vcc(nge as NotI1(Ge) + cmpneg)'] += 1
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
                if x.startswith('op_sel_hi:'):
                    bits = [int(y) for y in x.split(':')[1].strip('[]').split(',')]
                    opsel = sum(bit << i for i, bit in enumerate(bits))
            sc = f'Some(MmaScale {{ sa: {self.src(t[4])}, sb: {self.src(t[5])}, opsel: {opsel} }})'
        kind = f'MmaKind {{ shape: {shape}, a: {ta}, b: {tb}, scaled: {"true" if scaled else "false"} }}'
        self.dst(t[0], 'Frag16', f'Mfma {{ kind: {kind}, a: {a}, b: {b}, c: {c}, scale: {sc} }}')
        return True

    def h_v_mfma_scale_f32_32x32x64_f8f6f4(self, t, m, op): return self.mfma(t, m, True, 'M32N32K64', 'Fp8', 'Fp8')
    def h_v_mfma_f32_32x32x64_f8f6f4(self, t, m, op): return self.mfma(t, m, False, 'M32N32K64', 'Fp8', 'Fp8')

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
        self.dst(t[0], 'I32', f'Select {{ ty: I32, cond: Val({c}), a: {self.src(t[2], "I32")}, b: {self.src(t[1], "I32")} }}')
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
            return False
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
            self.m0 = (self.src(t[1]), self.src(t[2]))
            self.notes['m0 write (set_m0 in exact mode)'] += 1
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
            self.notes['m0 write (set_m0 in exact mode)'] += 1
            return True
        i = imm(t[1])
        if i is not None:
            self.dst(t[0], 'I32', f'ConstI32({i if i < 2**31 else i - 2**32})')
            return True
        try:
            x = float(t[1])
            self.dst(t[0], 'F32', f'ConstF32({x!r})')
            return True
        except ValueError:
            pass
        s = self.src(t[1], 'I32')
        self.dst(t[0], 'U32', f'AddI32({s}, I32(0))')
        self.notes['s_mov as AddI32(x, 0) (needs a constant / copy form)'] += 1
        return True

    def h_s_addk_i32(self, t, m, op):
        s = self.src(t[0])
        self.dst(t[0], 'U32', f'AddI32({s}, {self.src(t[1])})')
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
        self.dst(t[0], 'I32', f'Select {{ ty: I32, cond: Val({c}), a: {self.src(t[1], "I32")}, b: {self.src(t[2], "I32")} }}')
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
        m0 = getattr(self, 'm0', None)
        lds = 'I32(0)'
        if m0:
            v = self.nv
            self.nv += 1
            self.lines.append(f'%v{v}: I32 = AddI32({m0[0]}, {m0[1]})')
            lds = f'Val({v})'
        self.stmt(f'BufferLoadLds {{ rsrc: {rsrc}, voff: Some({voff}), soff: {soff}, lds_addr: {lds}, imm: {off}, dwords: {dw} }}')
        return True

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
        if vm:
            self.stmt(f'WaitVm {{ count: {vm.group(1)} }}')
        if lg:
            self.stmt(f'WaitLds {{ count: {lg.group(1)} }}')
        self.notes['waitcnt (waitcnt(vm, lgkm) in exact mode)'] += 1
        return True

    def h_s_nop(self, t, m, op):
        self.stmt(f'# s_nop {t[0]} (exact mode: s_nop(n))')
        self.notes['s_nop (exact mode: s_nop(n))'] += 1
        return True

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
            if op == 's_add_u32' and i + 1 < len(ins) and ins[i + 1][1] == 's_addc_u32':
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
            self.lift(op, args)
            i += 1


def main():
    path = sys.argv[1]
    secs = sys.argv[sys.argv.index('--sections') + 1].split(',')
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'lifted'
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
            f.write(f'kernarg_size 8  workgroup 512  lds 163840  value_count {L.nv}\n')
            f.write('profile None  lower_opts []  sched_opts None\n')
            f.write('var_tys []\n')
            f.write('arg Arg { name: "p", ty: Ptr, kernarg_offset: 0, size: 8 }\n')
            for p in L.pins:
                f.write(p + '\n')
            defined = set(int(x) for x in re.findall(r'^%v(\d+):', '\n'.join(L.shell + L.lines), re.M))
            gaps = [i for i in range(L.nv) if i not in defined]
            if gaps:
                f.write('values { ' + ' '.join(f'%v{i}: I32' for i in gaps) + ' }\n')
            f.write('body {\n')
            for s in L.shell + L.lines:
                f.write('  ' + s + '\n')
            f.write('}\n')


if __name__ == '__main__':
    main()
