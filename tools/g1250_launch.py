#!/usr/bin/env python3
"""Hang-safe single-kernel launch on gfx1250 (the replica harness front end; runs ON the node, inside the caller's GPU lock).

Checks before launch (refuses on any failure, never "tries"):
  - the kernel's metadata (llvm-readelf --notes): kernarg segment size == the blob size, block <= max flat workgroup size, LDS;
  - cluster requirement: a kernel whose code uses cluster barriers (`s_barrier_signal -3` / `s_barrier_wait 0xfffd`), reads the cluster id
    registers (ttmp6) or issues TDM loads is a CLUSTER kernel unless --no-cluster-ok is given with a reason; it is launched only with explicit
    cluster dims (--cluster X,Y,Z). A cluster kernel launched 1x1 or with wrong dims can hang the GPU (TDM cluster loads outside a cluster are not
    downgraded on MI450 / MI430; cluster barriers need a cluster id);
  - the run is wrapped in `timeout -s KILL` (SIGTERM is not enough for a wedged process); the host itself polls with a deadline.
Only the PID this script starts is ever signalled.

    python3 tools/g1250_launch.py --co K.co --kernel NAME --kernarg ka.bin --grid X,Y,Z --block X,Y,Z [--cluster X,Y,Z] [--lds N]
        [--deadline-ms 5000] [--iters 1] [--host build/g1250_launch_host] [--dis K.dis]
"""
import argparse, os, re, subprocess, sys

ROCM = os.environ.get('ROCM_PATH', '/opt/rocm')


def metadata(co, kernel):
    p = subprocess.run([os.path.join(ROCM, 'llvm/bin/llvm-readelf'), '--notes', co], capture_output=True, text=True, timeout=60)
    t = p.stdout
    i = t.find(f'.name:           {kernel}') if f'.name:           {kernel}' in t else t.find(kernel)
    if i < 0:
        sys.exit(f'REFUSE: kernel {kernel} not in the metadata of {co}')
    # the kernel's block of the YAML (the keys precede / follow .name; take the whole kernel entry around it)
    start = t.rfind('  - .', 0, i)
    end = t.find('\n  - ', i)
    blk = t[start:end if end > 0 else None]
    get = lambda k: (re.search(rf'\.{k}:\s+(\d+)', blk) or [None, None])[1]
    return dict(kernarg=int(get('kernarg_segment_size') or -1), lds=int(get('group_segment_fixed_size') or 0),
                max_block=int(get('max_flat_workgroup_size') or 0), wave=int(get('wavefront_size') or 0))


def needs_cluster(dis_text):
    reasons = []
    if re.search(r's_barrier_signal\s+-3|s_barrier_wait\s+0xfffd', dis_text):
        reasons.append('cluster barrier')
    if re.search(r'\bttmp6\b', dis_text):
        reasons.append('reads ttmp6 (cluster ids)')
    if re.search(r'tensor_load_to_lds|\.long 0x[dD]031', dis_text):
        reasons.append('TDM loads (cluster multicast possible)')
    return reasons


def triple(s):
    v = [int(x) for x in s.split(',')]
    if len(v) != 3:
        raise argparse.ArgumentTypeError('X,Y,Z')
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--co', required=True)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--kernarg', required=True)
    ap.add_argument('--grid', type=triple, required=True)
    ap.add_argument('--block', type=triple, required=True)
    ap.add_argument('--cluster', type=triple, default=[0, 0, 0])
    ap.add_argument('--lds', type=int, default=0, help='dynamic LDS bytes')
    ap.add_argument('--deadline-ms', type=int, default=5000)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--host', default='build/g1250_launch_host')
    ap.add_argument('--dis', help='disassembly of the code object (else produced with llvm-objdump)')
    ap.add_argument('--no-cluster-ok', help='reason why a kernel with cluster features may run without a cluster')
    a = ap.parse_args()

    md = metadata(a.co, a.kernel)
    ka = os.path.getsize(a.kernarg)
    if md['kernarg'] != ka:
        sys.exit(f"REFUSE: kernarg blob {ka} bytes, the kernel declares {md['kernarg']}")
    nb = a.block[0] * a.block[1] * a.block[2]
    if md['max_block'] and nb > md['max_block']:
        sys.exit(f"REFUSE: block {nb} > max_flat_workgroup_size {md['max_block']}")
    dis = open(a.dis).read() if a.dis else subprocess.run([os.path.join(ROCM, 'llvm/bin/llvm-objdump'), '-d', '--mcpu=gfx1250', a.co],
                                                          capture_output=True, text=True, timeout=300).stdout
    why = needs_cluster(dis)
    has_cluster = any(a.cluster)
    if why and not has_cluster and not a.no_cluster_ok:
        sys.exit(f"REFUSE: the kernel needs a cluster launch ({', '.join(why)}); pass --cluster with the dims it was built for")
    if has_cluster and any(g % c for g, c in zip(a.grid, a.cluster)):
        sys.exit('REFUSE: cluster dims must divide the grid')
    kill_after = max(10, a.deadline_ms // 1000 + 10)
    cmd = ['timeout', '-s', 'KILL', str(kill_after), a.host, a.co, a.kernel, a.kernarg, str(md['kernarg'])] + [str(x) for x in a.grid + a.block + a.cluster] \
        + [str(a.lds), str(a.deadline_ms), str(a.iters)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    sys.stdout.write(out)
    if p.returncode == 137:
        print(f'LAUNCH_FAIL killed after {kill_after} s (hang): STOP, check the GPU / dmesg before any further run')
    sys.exit(p.returncode)


if __name__ == '__main__':
    main()
