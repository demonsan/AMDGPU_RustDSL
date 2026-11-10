"""L1 op graphs (docs/l1-graph.md): plan a model-level op graph with the fusion rules (``amdc graph plan``) and run the plan through the
existing op wrappers.

    g = {"tensors": {...}, "dims": {"M": 256}, "ops": [...], "outputs": [...]}
    p = aj.graph.plan(g)                  # {"launches": [...], "rejected": [...], "errors": [...]}
    aj.graph.run(p, {"x": x, "g": gamma, "q": q, "s": scale})

The plan is per concrete shape and cached by the graph's JSON text. Float attributes are JSON strings (``"eps": "1e-6"``).
"""
import inspect
import json
import subprocess

from ._compile import amdc_path

_plans = {}


class PlanError(ValueError):
    """The graph has ops no kernel covers (the plan's ``errors``)."""


def plan(graph, *, cus=None, fuse=True):
    """The launch plan of ``graph`` (a dict or a JSON string): ``amdc graph plan`` (cached). ``fuse=False``: every op through its single-op
    rule (the unfused reference plan)."""
    text = graph if isinstance(graph, str) else json.dumps(graph, sort_keys=True)
    key = (text, cus, fuse)
    hit = _plans.get(key)
    if hit is None:
        args = [amdc_path(), "graph", "plan", "-"]
        if cus is not None:
            args += ["--cus", str(int(cus))]
        if not fuse:
            args.append("--no-fuse")
        res = subprocess.run(args, input=text, capture_output=True, text=True)
        if res.returncode not in (0, 1) or not res.stdout.strip():
            raise ValueError(f"amdc graph plan failed: {res.stderr.strip()}")
        hit = _plans[key] = json.loads(res.stdout)
    return hit


# parameters that carry floats as JSON strings
_FLOATS = {"eps", "scale", "k_scale", "v_scale"}


def _wrapper(name):
    from . import ops, ops_compute, ops_mem, ops_more
    for mod in (ops_mem, ops_compute, ops, ops_more):
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    raise ValueError(f"graph: no wrapper `{name}`")


def run(p, tensors, *, stream=None, check=True):
    """Executes plan ``p`` (from ``plan``) with ``tensors`` (graph tensor name -> buffer). Each launch calls its Python wrapper (``py``) with
    the bound tensors and the plan's parameters (the ones the wrapper takes). Returns the list of the wrappers' results."""
    if p.get("errors"):
        raise PlanError("; ".join(f"op {e['op']}: {e['why']}" for e in p["errors"]))
    out = []
    for l in p["launches"]:
        fn = _wrapper(l["py"])
        sig = inspect.signature(fn).parameters
        kw = {}
        for role, t in l["bind"].items():
            if t not in tensors:
                raise KeyError(f"graph: tensor `{t}` ({l['py']}.{role}) not given")
            kw[role] = tensors[t]
        for k, v in l["params"].items():
            if k in sig:
                kw[k] = float(v) if k in _FLOATS and isinstance(v, str) else v
        if "stream" in sig:
            kw["stream"] = stream
        if "check" in sig:
            kw["check"] = check
        out.append(fn(**kw))
    return out


def kernels(p):
    """The kernel names of a plan, in launch order."""
    return [k for l in p["launches"] for k in l["kernels"]]
