"""Pre-warm the JIT disk cache offline (docs/compile-time.md): compile (and certify) a list of kernel shapes in parallel so the first launch of each is a disk hit.

    AMDGPU_JIT_AMDC=/path/amdc python3 -m amdgpu_jit.prewarm shapes.tsv [--workers N]

`shapes.tsv`: one `kernel<TAB>name=value,name=value` per line (`#` comments; an empty spec is the generic kernel). No GPU is needed: only the compile, the certificate and the cache entry are produced,
nothing is loaded. The same entries serve every later process that runs with the same compiler, toolchain and cache directory (`AMDGPU_JIT_CACHE_DIR`)."""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import _compile as C


def _val(s):
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def parse_shapes(text):
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        k, _, spec = line.partition("\t")
        out.append((k.strip(), {a.partition("=")[0].strip(): _val(a.partition("=")[2].strip()) for a in spec.split(",") if a.strip()}))
    return out


def prewarm(shapes, workers=None):
    """Compiles `shapes` ([(kernel, spec dict)]) into the disk cache on `workers` threads; returns (shapes, seconds, compiles, disk hits)."""
    before = dict(C.stats)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers or os.cpu_count() or 8) as ex:
        list(ex.map(lambda ks: C._get_kernel(ks[0], ks[1], None, load=False), shapes))
    return len(shapes), time.perf_counter() - t0, C.stats["compiles"] - before.get("compiles", 0), C.stats["disk_hits"] - before.get("disk_hits", 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    with open(a.file) as f:
        shapes = parse_shapes(f.read())
    n, dt, compiled, hits = prewarm(shapes, a.workers)
    print(f"prewarm: {n} shapes in {dt:.2f} s ({dt * 1e3 / max(n, 1):.2f} ms each): {compiled} compiled, {hits} already cached")
    C.close_servers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
