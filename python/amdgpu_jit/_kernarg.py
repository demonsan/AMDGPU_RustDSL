"""Kernarg packing and host-contract validation from the ``amdc jit-info`` JSON (schema 1, docs/python-jit.md).

The JSON lists every byte of the kernarg segment (``args``, padding included) and the host contract the GPU-semantics verifier proved the
kernel against (launch grid, kernarg relations / ranges / sets, tensor allocation sizes, free symbols), as expression trees over the kernarg
scalars. ``pack`` fills the segment by argument name; ``validate`` evaluates every contract item for one concrete launch and raises
``ContractError`` naming the violated item - a launch that passes is inside what the kernel was verified for.
"""
import struct

SCHEMA = 1


class ContractError(ValueError):
    pass


def eval_ext(e, ka, grid, syms):
    """Evaluates an expression tree: {"c"} | {"ka": offset} | {"grid": axis} | {"sym": name} | {"add"|"sub"|"mul": [E, E]} |
    {"ceil_div"|"floor_div": [E, k]}."""
    if "c" in e:
        return int(e["c"])
    if "ka" in e:
        off = int(e["ka"])
        if off not in ka:
            raise ContractError(f"contract reads kernarg offset 0x{off:x}, which is not a scalar argument")
        return ka[off]
    if "grid" in e:
        return int(grid[int(e["grid"])])
    if "sym" in e:
        n = e["sym"]
        if n not in syms:
            raise ContractError(f"contract symbol {n} not supplied")
        return int(syms[n])
    for op in ("add", "sub", "mul", "ceil_div", "floor_div"):
        if op in e:
            a, b = e[op]
            x = eval_ext(a, ka, grid, syms)
            y = eval_ext(b, ka, grid, syms) if isinstance(b, dict) else int(b)
            if op == "add":
                return x + y
            if op == "sub":
                return x - y
            if op == "mul":
                return x * y
            if y <= 0:
                raise ContractError(f"{op} by non-positive {y}")
            return -((-x) // y) if op == "ceil_div" else x // y
    raise ContractError(f"unknown expression node {e}")


def fmt_ext(e):
    """Infix rendering for messages."""
    if "c" in e:
        return str(e["c"])
    if "ka" in e:
        return f"ka(0x{int(e['ka']):x})"
    if "grid" in e:
        return f"grid({e['grid']})"
    if "sym" in e:
        return e["sym"]
    for op, s in (("add", "+"), ("sub", "-"), ("mul", "*")):
        if op in e:
            return f"({fmt_ext(e[op][0])}{s}{fmt_ext(e[op][1])})"
    for op, s in (("ceil_div", "ceil"), ("floor_div", "floor")):
        if op in e:
            a, b = e[op]
            return f"{s}({fmt_ext(a)}/{b if not isinstance(b, dict) else fmt_ext(b)})"
    return str(e)


class KernelInfo:
    """Parsed ``amdc jit-info`` object."""

    def __init__(self, d):
        if d.get("schema") != SCHEMA:
            raise ValueError(f"jit-info schema {d.get('schema')} unsupported (expected {SCHEMA})")
        self.raw = d
        self.kernel = d["kernel"]
        self.symbol = d["symbol"]
        self.block = tuple(d["block"])
        self.kernarg_size = int(d["kernarg_size"])
        self.args = sorted(d["args"], key=lambda a: a["offset"])
        self.by_name = {a["name"]: a for a in self.args}
        self.contract = d.get("contract", {})
        self.specialize = d.get("specialize", {}) or {}
        end = 0
        for a in self.args:
            if a["offset"] < end:
                raise ValueError(f"{self.kernel}: argument {a['name']} overlaps the previous one")
            end = a["offset"] + a["size"]
        if end > self.kernarg_size:
            raise ValueError(f"{self.kernel}: arguments end at {end} > kernarg_size {self.kernarg_size}")

    def tensor_args(self):
        return [a["name"] for a in self.args if a["kind"] == "ptr"]


def pack(info: KernelInfo, values: dict):
    """Packs the kernarg segment. ``values``: name -> int (pointers / integers) or float (f32). Every non-padding argument must be given;
    a specialized argument must equal its specialized value (the kernel uses the constant). Returns (bytes, {offset: u32 value})."""
    buf = bytearray(info.kernarg_size)
    ka = {}
    unknown = set(values) - set(info.by_name)
    if unknown:
        raise ContractError(f"{info.kernel}: unknown argument(s) {sorted(unknown)}")
    for a in info.args:
        kind, name, off = a["kind"], a["name"], a["offset"]
        if kind == "pad":
            continue
        if name not in values:
            raise ContractError(f"{info.kernel}: missing argument {name}")
        v = values[name]
        spec = a.get("specialized")
        if spec is not None and v != spec:
            raise ContractError(f"{info.kernel}: argument {name} = {v}, but the kernel is specialized for {spec}")
        if kind == "ptr":
            struct.pack_into("<Q", buf, off, int(v))
        elif kind == "i32":
            if not -(1 << 31) <= int(v) < (1 << 31):
                raise ContractError(f"{name} = {v} does not fit i32")
            struct.pack_into("<i", buf, off, int(v))
            ka[off] = int(v) & 0xFFFFFFFF
        elif kind == "u32":
            if not 0 <= int(v) < (1 << 32):
                raise ContractError(f"{name} = {v} does not fit u32")
            struct.pack_into("<I", buf, off, int(v))
            ka[off] = int(v)
        elif kind == "f32":
            struct.pack_into("<f", buf, off, float(v))
            ka[off] = struct.unpack_from("<I", buf, off)[0]
        else:
            raise ContractError(f"{info.kernel}: argument {name} of unsupported kind {kind}")
    # specialized args keep their value in the contract as constants already; also expose them as ka for expressions that still read them
    return bytes(buf), ka


def validate(info: KernelInfo, ka: dict, grid, syms: dict, sizes: dict):
    """Checks one launch against the contract. ``sizes``: pointer argument name -> addressable bytes."""
    c = info.contract
    grid = (tuple(grid) + (1, 1, 1))[:3]
    for s in c.get("syms", []):
        if s["name"] not in syms:
            raise ContractError(f"{info.kernel}: contract symbol {s['name']} not supplied")
        v = syms[s["name"]]
        if v < s["lo"] or (s.get("hi") is not None and v > s["hi"]):
            raise ContractError(f"{info.kernel}: symbol {s['name']} = {v} outside [{s['lo']}, {s.get('hi')}]")
    for g in c.get("grid", []):
        want = eval_ext(g["ext"], ka, grid, syms)
        if grid[g["axis"]] != want:
            raise ContractError(f"{info.kernel}: grid[{g['axis']}] = {grid[g['axis']]}, the contract requires {fmt_ext(g['ext'])} = {want}")
    for r in c.get("ka_is", []):
        want = eval_ext(r["ext"], ka, grid, syms)
        have = ka.get(r["offset"])
        if have != want:
            raise ContractError(f"{info.kernel}: kernarg 0x{r['offset']:x} = {have}, the contract requires {fmt_ext(r['ext'])} = {want}")
    for r in c.get("ka_range", []):
        have = ka.get(r["offset"])
        if have is None or have < r["lo"] or (r.get("hi") is not None and have > r["hi"]):
            raise ContractError(f"{info.kernel}: kernarg 0x{r['offset']:x} = {have} outside [{r['lo']}, {r.get('hi')}]")
    for r in c.get("ka_set", []):
        have = ka.get(r["offset"])
        if have not in r["values"]:
            raise ContractError(f"{info.kernel}: kernarg 0x{r['offset']:x} = {have} not in {r['values']}")
    for t in c.get("tensors", []):
        need = eval_ext(t["alloc_bytes"], ka, grid, syms)
        have = sizes.get(t["arg"])
        if have is None:
            raise ContractError(f"{info.kernel}: no buffer size for {t['arg']}")
        if have < need:
            raise ContractError(f"{info.kernel}: {t['arg']} has {have} bytes, the contract requires {fmt_ext(t['alloc_bytes'])} = {need}")
