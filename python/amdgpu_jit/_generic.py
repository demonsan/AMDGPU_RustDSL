"""Generic launch driven by ``amdc jit-info``: every registered kernel is callable by name with its tensors and scalars.

``launch(kernel, args)`` packs the kernarg segment from the jit-info argument table (by name, or positionally: pointer arguments in offset
order, then scalar arguments in offset order), derives the contract's free symbols from the scalar relations (``ka_is``) and the launch grid
from the contract's grid expressions, checks the launch against the verified host contract (allocation sizes included) and launches. The ops
in ``ops.py`` / ``ops_mem.py`` are thin wrappers that pick the kernel name for a shape and pass the tensors.
"""
from ._compile import get_kernel
from ._kernarg import ContractError, eval_ext, scalars_to_ka, pack_fast
from ._tensor import as_ref, current_stream


def _is_tensor(v):
    return not isinstance(v, (int, float, bool))


def _unknowns(e, known):
    if not isinstance(e, dict):
        return set()
    if "sym" in e:
        return set() if e["sym"] in known else {e["sym"]}
    out = set()
    for op in ("add", "sub", "mul", "ceil_div", "floor_div"):
        if op in e:
            for x in e[op]:
                out |= _unknowns(x, known)
    return out


def _solve(e, target, ka, grid, syms):
    """Solves ``e == target`` for the single unknown symbol of e when e is linear in it (sym, + - * by known terms). Returns (name, value) or
    None (not invertible / not exact)."""
    if "sym" in e:
        return e["sym"], target
    for op in ("add", "sub", "mul"):
        if op in e:
            a, b = e[op]
            ua, ub = _unknowns(a, syms), _unknowns(b, syms) if isinstance(b, dict) else set()
            if ua and not ub:
                kb = eval_ext(b, ka, grid, syms) if isinstance(b, dict) else int(b)
                t = {"add": target - kb, "sub": target + kb}.get(op)
                if op == "mul":
                    if kb == 0 or target % kb:
                        return None
                    t = target // kb
                return _solve(a, t, ka, grid, syms)
            if ub and not ua:
                ka_ = eval_ext(a, ka, grid, syms)
                t = {"add": target - ka_, "sub": ka_ - target}.get(op)
                if op == "mul":
                    if ka_ == 0 or target % ka_:
                        return None
                    t = target // ka_
                return _solve(b, t, ka, grid, syms)
    return None


def infer_syms(info, ka, grid=None, syms=None):
    """The contract symbols implied by the scalar arguments (and the grid when given): ``ka_is`` / grid relations with one unknown symbol,
    to a fixpoint. Symbols already in ``syms`` are kept."""
    syms = dict(syms or {})
    c = info.contract
    eqs = [(x["ext"], ka.get(x["offset"])) for x in c.get("ka_is", []) if ka.get(x["offset"]) is not None]
    if grid is not None:
        g = (tuple(grid) + (1, 1, 1))[:3]
        eqs += [(x["ext"], g[int(x["axis"])]) for x in c.get("grid", [])]
    changed = True
    while changed:
        changed = False
        for e, v in eqs:
            # kernarg values are u32 in `ka`: read them back as signed for the relations
            v = v - (1 << 32) if v >= (1 << 31) else v
            if len(_unknowns(e, syms)) == 1:
                r = _solve(e, v, ka, grid or (1, 1, 1), syms)
                if r is not None:
                    syms[r[0]] = r[1]
                    changed = True
    return syms


def contract_grid(info, ka, syms):
    """The launch grid the contract prescribes (axes it does not constrain: 1)."""
    g = [1, 1, 1]
    for x in info.contract.get("grid", []):
        g[int(x["axis"])] = eval_ext(x["ext"], ka, (1, 1, 1), syms)
    return tuple(g)


def launch(kernel, args=None, *, ptrs=None, scalars=None, grid=None, syms=None, spec=None, stream=None, check=True):
    """Launches ``kernel`` (a registered name). Arguments either by name (``args``: name -> tensor-like or number) or positionally
    (``ptrs``: tensor-likes for the pointer arguments in offset order, ``scalars``: numbers for the scalar arguments in offset order).
    ``grid`` / ``syms`` default to what the contract implies. Returns the Kernel."""
    kern = get_kernel(kernel, spec)
    info = kern.info
    named = dict(args or {})
    if ptrs is not None or scalars is not None:
        pnames = [a["name"] for a in info.args if a["kind"] == "ptr"]
        snames = [a["name"] for a in info.args if a["kind"] not in ("ptr", "pad")]
        if ptrs is not None:
            if len(ptrs) != len(pnames):
                raise ContractError(f"{kernel}: {len(ptrs)} tensors for the pointer arguments {pnames}")
            named.update(zip(pnames, ptrs))
        if scalars is not None:
            if len(scalars) != len(snames):
                raise ContractError(f"{kernel}: {len(scalars)} scalars for the scalar arguments {snames}")
            named.update(zip(snames, scalars))
    vals, sizes, refs = {}, {}, []
    for n, v in named.items():
        a = info.by_name.get(n)
        if a is None:
            raise ContractError(f"{kernel}: unknown argument {n} (arguments: {[x['name'] for x in info.args if x['kind'] != 'pad']})")
        if a["kind"] == "ptr":
            if isinstance(v, int):
                vals[n] = v  # a raw pointer: no size (the contract check needs `sizes` - pass a RawPtr instead)
                continue
            r = as_ref(v)
            refs.append(r)
            vals[n], sizes[n] = r.ptr, r.cap
        else:
            vals[n] = float(v) if a["kind"] == "f32" else int(v)
    _, sc = pack_fast(info, vals)
    ka = scalars_to_ka(info, sc)
    syms = infer_syms(info, ka, grid, syms)
    if grid is None:
        try:
            grid = contract_grid(info, ka, syms)
        except ContractError as e:
            raise ContractError(f"{kernel}: cannot derive the launch grid ({e}); pass grid= / syms= (the scalar arguments must satisfy "
                                f"the kernel's relations, e.g. row counts a multiple of the rows per CTA)") from None
    kern.launch(vals, grid, syms, sizes, current_stream(refs, stream), check)
    return kern
