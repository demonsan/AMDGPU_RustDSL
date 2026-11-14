#!/usr/bin/env python3
"""Same-session performance A/B of two commits over every integrated kernel family (docs/perf-ab.md).

    python3 tools/perf_ab.py --old <git-ref> --new <git-ref|.> [--families a,b] [--only REGEX] [--reps 5] [--floor 0.02] [--k 2.0] [--out DIR]
    (wrapper: bash tools/perf_ab.sh, which sets the node environment)

1. builds `amdc` of both commits (git worktrees under build/ab/trees, cargo target per commit), 2. emits and assembles the .co of every kernel the cases name, once per commit,
3. builds the GPU hosts (tests/gpu/*_host.cpp of the NEW tree; the same host binary runs both sides), 4. runs every case INTERLEAVED in one session: per repetition, per case,
old and new back to back (order alternating per repetition), the vendor side (where a case has one) right after, 5. the noise model and the verdict (below), 6. `report.md`, `report.json`.

Noise model (per case): samples of both sides are converted to a performance number (higher is better; a time is inverted); `ratio = median(new) / median(old)`;
each side's scatter is the robust relative sigma `1.4826 * MAD / median`; the allowed drop is `thr = max(--floor, --k * hypot(sigma_old, sigma_new))`;
the case FAILS when `ratio < 1 - thr` (or when a correctness marker of the host is missing on the new side), is IMPROVED when `ratio > 1 + thr`.
Family verdict: the geometric mean of the case ratios must not fall below `1 - max(--family-floor, family noise)`. The overall verdict is PASS iff no case and no family fails.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, re, shlex, statistics, subprocess, sys, time
from dataclasses import dataclass, field

if __name__ == '__main__':  # the case library imports `perf_ab`: it must see THIS module's classes
    sys.modules['perf_ab'] = sys.modules['__main__']

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# case model (the case library is tools/perf_ab_cases.py)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
METRICS = {
    'tflops': (r'([0-9]+(?:\.[0-9]+)?)\s*TF(?:LOP)?/?s', 'high'),
    'gbs': (r'([0-9]+(?:\.[0-9]+)?)\s*GB/s', 'high'),
    'ms': (r'([0-9]+(?:\.[0-9]+)?)\s*ms/iter', 'low'),
    'us': (r'([0-9]+(?:\.[0-9]+)?)\s*us\b', 'low'),
}


@dataclass
class Run:
    """What one side executes: the kernels to emit, the command (tokens with {co:NAME} / {codir} / {host:NAME}), its environment."""
    emit: list
    cmd: list
    env: dict = field(default_factory=dict)


@dataclass
class Case:
    id: str
    family: str
    build: object                      # callable(Side) -> Run   (may call side.dispatch(...) to ask that commit's launch rules)
    metric: str = 'tflops'             # key of METRICS, or a regex with one group (then `better` applies)
    better: str = None
    line: str = None                   # regex: only the output lines matching it are searched for the metric
    ok: str = None                     # regex that must match the output (correctness marker of the host)
    fail: str = r'\b[A-Z0-9]+_FAIL\b|\bFAIL(?:ED)?\b'   # regex that must NOT match (a host's failure marker: XYZ_FAIL / FAIL) - a wrong result fails the case
                                       # even when a weak `ok` pattern also matches (topk_per_row: `past_end_bad=0 ... TKR_FAIL` matched `bad*=0`)
    vendor: object = None              # callable() -> Run (no emit), the vendor side, same session
    vendor_line: str = None            # regex of the vendor lines (with `vendor_same`: lines of the SAME output as ours, hosts that run both kernels in one process)
    vendor_same: bool = False
    quick: bool = False                # member of the --quick set (the per-merge gate)
    parts: list = None                 # several metrics of one run, each judged on its own: [dict(name=, metric=, line=, vendor_line=, better=)]
    floor: float = None                # per-case floor for the allowed drop (default --floor)
    timeout: int = 600


class Side:
    """One commit: its tree, its amdc, its output dir."""

    def __init__(self, name, ref, sha, tree, out):
        self.name, self.ref, self.sha, self.tree = name, ref, sha, tree
        self.dir = os.path.join(out, name)
        self.codir = os.path.join(self.dir, 'co')
        self.amdc = None
        self.missing = {}
        self._disp = {}
        self.emit_env = {}
        self.fallback = None   # a commit whose launch rules answer a `dispatch` subcommand this commit does not have yet

    def dispatch(self, *args):
        """`amdc dispatch ...` of this commit as a dict (the launch rules of the commit under test)."""
        key = tuple(str(a) for a in args)
        if key not in self._disp:
            p = subprocess.run([self.amdc, 'dispatch', *key], capture_output=True, text=True, cwd=self.tree)
            if p.returncode != 0 and self.fallback is not None and 'usage:' in p.stderr:
                self._disp[key] = self.fallback.dispatch(*key)
                return self._disp[key]
            if p.returncode != 0:
                raise RuntimeError(f'{self.name}: amdc dispatch {" ".join(key)}: {p.stderr.strip()[:200]}')
            self._disp[key] = json.loads(p.stdout)
        return self._disp[key]


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def gpu_guard():
    """USER RULE: GPU tests only on GPUs 0-3 (never 4-7). Same rule as tools/gpu_guard.sh: HIP_VISIBLE_DEVICES (default 0) indexes into ROCR_VISIBLE_DEVICES when that is set; the physical GPUs selected must be 0-3."""
    os.environ.setdefault('HIP_VISIBLE_DEVICES', '0' if not os.environ.get('ROCR_VISIBLE_DEVICES') else '0')
    rocr = [x for x in os.environ.get('ROCR_VISIBLE_DEVICES', '').split(',') if x]
    hip = [x for x in os.environ.get('HIP_VISIBLE_DEVICES', '').split(',') if x]
    for v in rocr + hip:
        if not v.isdigit():
            sys.exit(f"gpu_guard: device list HIP='{os.environ.get('HIP_VISIBLE_DEVICES')}' ROCR='{os.environ.get('ROCR_VISIBLE_DEVICES', '')}' is not a plain number list: cannot prove it is 0-3")
    for h in hip:
        if rocr and int(h) >= len(rocr):
            sys.exit(f"gpu_guard: HIP_VISIBLE_DEVICES='{os.environ['HIP_VISIBLE_DEVICES']}' indexes into ROCR_VISIBLE_DEVICES='{os.environ['ROCR_VISIBLE_DEVICES']}' ({len(rocr)} entries): index {h} is out of range, every launch would fail. Set only one of the two variables")
        phys = rocr[int(h)] if rocr else h
        if phys not in ('0', '1', '2', '3'):
            sys.exit(f"gpu_guard: HIP_VISIBLE_DEVICES='{os.environ['HIP_VISIBLE_DEVICES']}' selects physical GPU '{phys}': GPU tests may only run on GPUs 0-3 (never 4-7, on any node)")


def env_default(name, default):
    return os.environ.get(name, default)


ROCM = env_default('ROCM_PATH', '/shared/apps/ubuntu/opt/rocm-7.0.2')
CLANG = env_default('CLANG', '/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang')
HSA = env_default('AITER_HSA', '/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950')
REF_CO = env_default('REF_CO', '/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co')


def sh(cmd, cwd=None, env=None, timeout=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, errors='replace')
    if check and p.returncode != 0:
        raise RuntimeError(f'{" ".join(map(str, cmd))[:200]}: exit {p.returncode}\n{(p.stdout + p.stderr)[-1500:]}')
    return p


def git(*args, cwd=ROOT):
    return sh(['git', *args], cwd=cwd, check=True).stdout.strip()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# build: trees, amdc, kernels, hosts
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def make_side(name, ref, out):
    if ref in ('.', 'WORKTREE'):
        sha = 'worktree'
        tree = ROOT
        try:
            sha = 'wt-' + git('rev-parse', '--short=8', 'HEAD')
        except Exception:
            pass
    else:
        sha = git('rev-parse', '--verify', ref + '^{commit}')
        tree = os.path.join(ROOT, 'build', 'ab', 'trees', sha[:12])
        if not os.path.isdir(tree):
            os.makedirs(os.path.dirname(tree), exist_ok=True)
            git('worktree', 'add', '--detach', tree, sha)
    return Side(name, ref, sha, tree, out)


def build_amdc(side, args):
    target = os.path.join(ROOT, 'build', 'ab', 'target', side.sha[:12])
    os.makedirs(target, exist_ok=True)
    env = dict(os.environ, AMD_CARGO_TARGET=target)
    profile = 'release' if args.release else 'debug'
    cmd = ['bash', 'tools/container/cargo.sh', 'build', '--bin', 'amdc'] + (['--release'] if args.release else [])
    t0 = time.time()
    p = sh(cmd, cwd=side.tree, env=env)
    if p.returncode != 0:
        raise RuntimeError(f'{side.name}: cargo build failed:\n{(p.stdout + p.stderr)[-2000:]}')
    side.amdc = os.path.join(target, profile, 'amdc')
    if not os.path.exists(side.amdc):
        raise RuntimeError(f'{side.name}: {side.amdc} not found after the build')
    print(f'[build] {side.name} {side.ref} -> {side.sha[:12]}: amdc in {time.time() - t0:.0f}s', flush=True)


def emit_kernel(side, k):
    co = os.path.join(side.codir, k + '.co')
    s = os.path.join(side.dir, 'asm', k + '.s')
    if os.path.exists(co) and os.path.getmtime(co) >= os.path.getmtime(side.amdc):
        return k, True, ''
    os.makedirs(os.path.dirname(s), exist_ok=True)
    p = sh([side.amdc, 'emit', k, '--backend', 'own', '--out', s], cwd=side.tree, timeout=1200, env=dict(os.environ, **side.emit_env))
    if p.returncode != 0 or not os.path.exists(s):
        return k, False, (p.stdout + p.stderr)[-300:]
    p = sh([CLANG, '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mcode-object-version=6', '-o', co, s], timeout=600)
    if p.returncode != 0:
        return k, False, 'assemble: ' + (p.stdout + p.stderr)[-300:]
    return k, True, ''


def emit_all(side, kernels, jobs):
    os.makedirs(side.codir, exist_ok=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(jobs) as ex:
        for k, ok, msg in ex.map(lambda k: emit_kernel(side, k), sorted(kernels)):
            if not ok:
                side.missing[k] = msg
    print(f'[emit] {side.name}: {len(kernels) - len(side.missing)}/{len(kernels)} kernels in {time.time() - t0:.0f}s' + (f' ({len(side.missing)} unavailable)' if side.missing else ''), flush=True)


def build_host(tree, name, out):
    src = os.path.join(tree, 'tests', 'gpu', name + '.cpp')
    exe = os.path.join(out, 'hosts', name)
    if not os.path.exists(src):
        return name, None, f'{src} missing'
    h = hashlib.sha1(open(src, 'rb').read()).hexdigest()[:10]
    stamp = exe + '.' + h
    if os.path.exists(exe) and os.path.exists(stamp):
        return name, exe, ''
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    cmd = [os.path.join(ROCM, 'bin', 'amdclang++'), '-x', 'hip', '-O2', '--offload-arch=gfx950', '--rocm-path=' + ROCM, '-std=c++17', '-I', os.path.join(tree, 'tests', 'gpu'),
           '-I', tree, src, '-o', exe, '-L' + os.path.join(ROCM, 'lib'), '-lamdhip64', '-Wl,-rpath,' + os.path.join(ROCM, 'lib')]
    p = sh(cmd, timeout=1200)
    if p.returncode != 0:
        return name, None, (p.stdout + p.stderr)[-400:]
    open(stamp, 'w').close()
    return name, exe, ''


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def expand(run, side, hosts):
    def sub(tok):
        tok = re.sub(r'\{co:([^}]+)\}', lambda m: os.path.join(side.codir, m.group(1) + '.co'), tok)
        tok = tok.replace('{codir}', side.codir)
        tok = re.sub(r'\{host:([^}]+)\}', lambda m: hosts.get(m.group(1)) or 'MISSING_HOST_' + m.group(1), tok)
        return tok
    return [sub(t) for t in run.cmd], {k: sub(str(v)) for k, v in run.env.items()}


def parse(text, metric, line):
    """(value, metric key) of the first metric of `metric` (a key / regex or a list of them, tried in order) found in the lines matching `line`."""
    keys = list(metric) if isinstance(metric, (list, tuple)) else [metric]
    lines = [l for l in text.splitlines() if (line is None or re.search(line, l))]
    for key in keys:
        pat = METRICS[key][0] if key in METRICS else key
        for l in lines:
            m = re.search(pat, l)
            if m:
                return float(m.group(1)), key
    return None, None


def run_once(cmd, env, timeout):
    e = dict(os.environ)
    e.update(env)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout, errors='replace')
        return p.stdout + p.stderr, p.returncode, time.time() - t0
    except subprocess.TimeoutExpired as ex:
        return (ex.stdout or '') if isinstance(ex.stdout, str) else '', -9, time.time() - t0


def mad_rel(xs):
    m = statistics.median(xs)
    if m == 0 or len(xs) < 3:
        return 0.0
    return 1.4826 * statistics.median([abs(x - m) for x in xs]) / m


def judge(old, new, better, floor, k):
    perf = (lambda v: v) if better == 'high' else (lambda v: 1.0 / v)
    po, pn = [perf(v) for v in old], [perf(v) for v in new]
    mo, mn = statistics.median(po), statistics.median(pn)
    ratio = mn / mo
    so, sn = mad_rel(po), mad_rel(pn)
    thr = max(floor, k * math.hypot(so, sn))
    verdict = 'FAIL' if ratio < 1 - thr else ('IMPROVED' if ratio > 1 + thr else 'OK')
    return dict(ratio=ratio, thr=thr, sigma_old=so, sigma_new=sn, med_old=statistics.median(old), med_new=statistics.median(new), verdict=verdict, n=min(len(old), len(new)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', required=True, help='git ref of the baseline commit')
    ap.add_argument('--new', required=True, help='git ref of the commit under test ("." = the working tree)')
    ap.add_argument('--families', default='', help='comma list of families (default: all); --list shows them')
    ap.add_argument('--only', default='', help='regex on the case id')
    ap.add_argument('--reps', type=int, default=None, help='interleaved repetitions per case (default 5, 3 with --quick; the noise model needs >= 3)')
    ap.add_argument('--floor', type=float, default=0.02, help='minimum allowed drop of a case (default 2 %%)')
    ap.add_argument('--k', type=float, default=2.0, help='noise multiplier: allowed drop = max(floor, k * hypot(sigma_old, sigma_new))')
    ap.add_argument('--family-floor', type=float, default=0.01, help='minimum allowed drop of a family geometric mean (default 1 %%)')
    ap.add_argument('--out', default='', help='output dir (default build/ab/run-<old>-<new>)')
    ap.add_argument('--jobs', type=int, default=8, help='parallel emit / assemble jobs')
    ap.add_argument('--release', action='store_true', help='release-profile amdc (default: debug)')
    ap.add_argument('--no-vendor', action='store_true', help='skip the vendor side of the cases')
    ap.add_argument('--list', action='store_true', help='list the cases and exit (no build)')
    ap.add_argument('--quick', action='store_true', help='the per-merge gate: the first and the last case of every group (kind / causal / host); minutes instead of hours')
    ap.add_argument('--old-emit-env', action='append', default=[], metavar='K=V', help='environment of the old side\'s `amdc emit` (experiment switches such as AMDC_SCHED_ADD); repeatable')
    ap.add_argument('--new-emit-env', action='append', default=[], metavar='K=V', help='environment of the new side\'s `amdc emit`; repeatable')
    ap.add_argument('--show', action='store_true', help='run each selected case once on the new side and print the command and the output tail (to write / debug a case); no session')
    ap.add_argument('--keep-going', action='store_true', help='do not stop on a host build failure')
    args = ap.parse_args()
    if args.reps is None:
        args.reps = 3 if args.quick else 5

    import perf_ab_cases as lib
    if not args.list:
        gpu_guard()
    out = os.path.abspath(args.out or os.path.join(ROOT, 'build', 'ab', f'run-{args.old.replace("/", "_")}-{args.new.replace("/", "_")}'.replace('.', 'wt')))
    os.makedirs(out, exist_ok=True)
    old, new = make_side('old', args.old, out), make_side('new', args.new, out)
    old.emit_env = dict(e.split('=', 1) for e in args.old_emit_env)
    new.emit_env = dict(e.split('=', 1) for e in args.new_emit_env)
    fam = [f for f in args.families.split(',') if f]
    cases = lib.cases(new.tree, args)
    cases = [c for c in cases if (not fam or c.family in fam) and (not args.only or re.search(args.only, c.id)) and (not args.quick or c.quick)]
    if args.list:
        for c in cases:
            print(f'{c.family:14s} {c.id}')
        print(f'{len(cases)} cases; families: ' + ', '.join(sorted({c.family for c in cases})))
        return 0
    if not cases:
        print('no case selected')
        return 2

    for s in (old, new):
        build_amdc(s, args)
    old.fallback, new.fallback = new, old   # a launch-rule subcommand only one commit has: the other commit's rule answers it (both sides then run the same plan)
    # the runs (they may ask each commit's dispatch rules), the kernels and the hosts they need
    runs = {}
    for c in cases:
        for s in (old, new):
            try:
                runs[(c.id, s.name)] = c.build(s)
            except Exception as e:  # a launch rule of that commit failed: the case cannot run there
                runs[(c.id, s.name)] = e
    kernels = {s.name: set() for s in (old, new)}
    hosts_needed = set()
    for (cid, sn), r in runs.items():
        if isinstance(r, Run):
            kernels[sn] |= set(r.emit)
            for t in r.cmd:
                hosts_needed |= set(re.findall(r'\{host:([^}]+)\}', t))
    for c in cases:
        if callable(c.vendor) and not args.no_vendor:
            for t in c.vendor().cmd:
                hosts_needed |= set(re.findall(r'\{host:([^}]+)\}', t))
    for s in (old, new):
        emit_all(s, kernels[s.name], args.jobs)
    hosts = {}
    with cf.ThreadPoolExecutor(min(4, max(1, len(hosts_needed)))) as ex:
        for name, exe, msg in ex.map(lambda h: build_host(new.tree, h, out), sorted(hosts_needed)):
            hosts[name] = exe
            if not exe:
                print(f'[host] {name}: build failed: {msg}', flush=True)
                if not args.keep_going:
                    return 2
    print(f'[host] {len(hosts)} hosts', flush=True)

    if args.show:
        for c in cases:
            r = runs[(c.id, 'new')]
            if not isinstance(r, Run):
                print(f'== {c.id}: no launch: {r}')
                continue
            cmd, env = expand(r, new, hosts)
            text, rc, dt = run_once(cmd, env, c.timeout)
            print(f'== {c.id}  rc={rc}  {dt:.1f}s\n   env {env}\n   cmd {" ".join(cmd)}')
            print('\n'.join('   | ' + l for l in text.strip().splitlines()[-int(os.environ.get("SHOW_LINES", "12")):]))
            if callable(c.vendor) and not args.no_vendor:
                vc, ve = expand(c.vendor(), new, hosts)
                vt, vrc, vdt = run_once(vc, ve, c.timeout)
                print(f'   vendor rc={vrc} {vdt:.1f}s: ' + ' '.join(vc))
                print('\n'.join('   | ' + l for l in vt.strip().splitlines()[-6:]))
        return 0

    # interleaved session
    def parts_of(c):
        return c.parts or [dict(name='', metric=c.metric, line=c.line, vendor_line=c.vendor_line, better=c.better)]
    samples = {}
    for c in cases:
        for pt in parts_of(c):
            samples[(c.id, pt['name'])] = {'old': [], 'new': [], 'vendor': [], 'ok_new': True, 'ok_old': True, 'notes': [], 'key': None}
    raw = open(os.path.join(out, 'raw.jsonl'), 'w')
    t0 = time.time()
    for rep in range(args.reps):
        for c in cases:
            order = (old, new) if rep % 2 == 0 else (new, old)
            for s_ in order:
                r = runs[(c.id, s_.name)]
                if not isinstance(r, Run):
                    for pt in parts_of(c):
                        samples[(c.id, pt['name'])]['notes'].append(f'{s_.name}: no launch ({r})')
                    continue
                miss = [k for k in r.emit if k in s_.missing]
                if miss:
                    for pt in parts_of(c):
                        samples[(c.id, pt['name'])]['notes'].append(f'{s_.name}: kernel unavailable: {miss[0]}')
                    continue
                cmd, env = expand(r, s_, hosts)
                text, rc, dt = run_once(cmd, env, c.timeout)
                good = (c.ok is None or re.search(c.ok, text, re.M) is not None) and (not c.fail or re.search(c.fail, text, re.M) is None) and rc == 0
                got = {}
                for pt in parts_of(c):
                    smp = samples[(c.id, pt['name'])]
                    v, key = parse(text, pt['metric'], pt['line'])
                    got[pt['name']] = v
                    smp['key'] = smp['key'] or key
                    if v is None or not good:
                        smp['ok_' + s_.name] = False
                        smp['notes'].append(f'{s_.name} rep {rep}: ' + ('no metric' if v is None else 'correctness marker missing or exit != 0') + f' (rc={rc})')
                    if v is not None:
                        smp[s_.name].append(v)
                    if c.vendor_same and s_ is new and pt.get('vendor_line') and not args.no_vendor:
                        vv, _ = parse(text, pt['metric'], pt['vendor_line'])
                        if vv is not None:
                            smp['vendor'].append(vv)
                raw.write(json.dumps(dict(case=c.id, side=s_.name, rep=rep, values=got, ok=good, rc=rc, sec=round(dt, 2), tail=text[-300:] if (not good or None in got.values()) else '')) + '\n')
            if callable(c.vendor) and not c.vendor_same and not args.no_vendor:
                vr = c.vendor()
                cmd, env = expand(vr, new, hosts)
                text, rc, dt = run_once(cmd, env, c.timeout)
                for pt in parts_of(c):
                    v, _ = parse(text, pt['metric'], pt.get('vendor_line') or c.vendor_line or pt['line'])
                    if v is not None:
                        samples[(c.id, pt['name'])]['vendor'].append(v)
        print(f'[run] repetition {rep + 1}/{args.reps} done ({time.time() - t0:.0f}s)', flush=True)
    raw.close()

    # verdict
    rows, fam_ratios = [], {}
    for c in cases:
      for pt in parts_of(c):
        smp = samples[(c.id, pt['name'])]
        cid = c.id + (('#' + pt['name']) if pt['name'] else '')
        better = pt.get('better') or (METRICS[smp['key']][1] if smp.get('key') in METRICS else 'high')
        row = dict(id=cid, family=c.family, better=better, notes=smp['notes'][:3])
        if len(smp['old']) >= 1 and len(smp['new']) >= 1:
            j = judge(smp['old'], smp['new'], better, c.floor if c.floor is not None else args.floor, args.k) if (len(smp['old']) >= 3 and len(smp['new']) >= 3) else None
            if j is None:  # too few samples for a noise estimate: a plain ratio with the floor
                perf = (lambda v: v) if better == 'high' else (lambda v: 1.0 / v)
                r = statistics.median([perf(v) for v in smp['new']]) / statistics.median([perf(v) for v in smp['old']])
                j = dict(ratio=r, thr=args.floor, sigma_old=0, sigma_new=0, med_old=statistics.median(smp['old']), med_new=statistics.median(smp['new']), verdict='FAIL' if r < 1 - args.floor else 'OK', n=min(len(smp['old']), len(smp['new'])))
            row.update(j)
            if not smp['ok_new']:
                row['verdict'] = 'FAIL'
                row['notes'].append('correctness marker missing on the new side')
            if smp['vendor']:
                vm = statistics.median(smp['vendor'])
                row['vendor'] = vm
                perf = (lambda v: v) if better == 'high' else (lambda v: 1.0 / v)
                row['new_vs_vendor'] = perf(j['med_new']) / perf(vm)
                row['old_vs_vendor'] = perf(j['med_old']) / perf(vm)
            fam_ratios.setdefault(c.family, []).append((j['ratio'], hypot2(j['sigma_old'], j['sigma_new'])))
        elif len(smp['new']) >= 1 and not smp['old']:
            row.update(verdict='NEW', med_new=statistics.median(smp['new']), ratio=None, thr=None)
        else:
            row.update(verdict='FAIL' if (any('new: ' in n for n in smp['notes']) or not smp['new']) else 'SKIP', ratio=None, thr=None)
            if row['verdict'] == 'FAIL' and not smp['notes']:
                row['notes'].append('no samples')
        rows.append(row)

    fam_rows = {}
    for f, rs in fam_ratios.items():
        g = math.exp(statistics.mean(math.log(r) for r, _ in rs))
        noise = math.sqrt(sum(s * s for _, s in rs)) / len(rs)
        thr = max(args.family_floor, args.k * noise)
        fam_rows[f] = dict(geomean=g, thr=thr, n=len(rs), verdict='FAIL' if g < 1 - thr else 'OK')
    fails = [r for r in rows if r['verdict'] == 'FAIL'] + [dict(id='family ' + f) for f, r in fam_rows.items() if r['verdict'] == 'FAIL']
    write_report(out, old, new, args, rows, fam_rows, fails)
    print(open(os.path.join(out, 'report.md')).read())
    return 1 if fails else 0


def hypot2(a, b):
    return math.hypot(a, b)


def fmt(v, nd=2):
    return '-' if v is None else (f'{v:.{nd}f}' if abs(v) < 1000 else f'{v:.0f}')


def write_report(out, old, new, args, rows, fam_rows, fails):
    L = [f'# perf A/B: old `{old.ref}` ({old.sha[:12]}) vs new `{new.ref}` ({new.sha[:12]})', '',
         f'{args.reps} interleaved repetitions per case, floor {args.floor:.0%}, k {args.k}, family floor {args.family_floor:.0%}; ratio = new / old in performance terms (> 1 = new is faster). Verdict: **{"FAIL" if fails else "PASS"}** ({len(fails)} failing).', '']
    L += ['## Families', '', '| family | cases | geomean new/old | allowed drop | verdict |', '|---|--:|--:|--:|---|']
    for f in sorted(fam_rows):
        r = fam_rows[f]
        L.append(f'| {f} | {r["n"]} | {r["geomean"]:.3f} | {r["thr"]:.1%} | {r["verdict"]} |')
    L += ['', '## Cases', '', '| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |', '|---|--:|--:|--:|--:|---|--:|--:|--:|']
    for r in sorted(rows, key=lambda r: (r['family'], r['id'])):
        L.append('| {id} | {o} | {n} | {ra} | {th} | {v} | {ve} | {nv} | {ov} |'.format(
            id=r['id'], o=fmt(r.get('med_old')), n=fmt(r.get('med_new')), ra=fmt(r.get('ratio'), 3), th=('-' if r.get('thr') is None else f'{r["thr"]:.1%}'), v=r['verdict'],
            ve=fmt(r.get('vendor')), nv=fmt(r.get('new_vs_vendor'), 3), ov=fmt(r.get('old_vs_vendor'), 3)))
    bad = [r for r in rows if r['verdict'] in ('FAIL', 'SKIP') and r['notes']]
    if bad:
        L += ['', '## Notes', '']
        for r in bad:
            L.append(f'- `{r["id"]}`: ' + '; '.join(r['notes']))
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(dict(old=old.sha, new=new.sha, rows=rows, families=fam_rows, verdict='FAIL' if fails else 'PASS'), open(os.path.join(out, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
