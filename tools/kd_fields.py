#!/usr/bin/env python3
"""Decode and compare the kernel descriptors (.kd, 64 bytes) of AMDGPU code objects (gfx9 layout).

usage: kd_fields.py A.co [B.co]     (prints the fields of A, or A vs B with the differing ones marked)
"""
import struct, subprocess, sys

RE = '/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/llvm-readelf'


def kd_bytes(path):
    out = subprocess.run([RE, '-s', '-S', '--wide', path], capture_output=True, text=True).stdout
    sym = [l.split() for l in out.splitlines() if l.strip().endswith('.kd')][0]
    val = int(sym[1], 16)
    data = open(path, 'rb').read()
    for l in out.splitlines():
        p = l.replace('[ ', '[').split()
        if len(p) > 5 and p[0].startswith('[') and p[0] != '[Nr]':
            try:
                addr, off, sz = int(p[3], 16), int(p[4], 16), int(p[5], 16)
            except ValueError:
                continue
            if sz and addr <= val < addr + sz:
                return data[off + val - addr: off + val - addr + 64]
    raise SystemExit(f'{path}: .kd not found')


def fields(b):
    w = struct.unpack('<16I', b)
    r1, r2, r3, props = w[12], w[13], w[11], w[14] & 0xffff
    return {
        'group_segment_fixed_size': w[0], 'private_segment_fixed_size': w[1], 'kernarg_size': w[2],
        'rsrc1.vgpr_blocks': r1 & 0x3f, 'rsrc1.sgpr_blocks': (r1 >> 6) & 0xf, 'rsrc1.priority': (r1 >> 10) & 3,
        'rsrc1.float_mode': (r1 >> 12) & 0xff, 'rsrc1.priv': (r1 >> 20) & 1, 'rsrc1.dx10_clamp': (r1 >> 21) & 1,
        'rsrc1.debug': (r1 >> 22) & 1, 'rsrc1.ieee': (r1 >> 23) & 1, 'rsrc1.bulky': (r1 >> 24) & 1, 'rsrc1.cdbg_user': (r1 >> 25) & 1,
        'rsrc1.fp16_ovfl': (r1 >> 26) & 1,
        'rsrc2.scratch_en': r2 & 1, 'rsrc2.user_sgpr': (r2 >> 1) & 0x1f, 'rsrc2.trap': (r2 >> 6) & 1, 'rsrc2.tgid_x': (r2 >> 7) & 1,
        'rsrc2.tgid_y': (r2 >> 8) & 1, 'rsrc2.tgid_z': (r2 >> 9) & 1, 'rsrc2.tg_info': (r2 >> 10) & 1, 'rsrc2.vgpr_workitem_id': (r2 >> 11) & 3,
        'rsrc2.excp_addr_watch': (r2 >> 13) & 1, 'rsrc2.excp_mem_viol': (r2 >> 14) & 1, 'rsrc2.lds_size': (r2 >> 15) & 0x1ff,
        'rsrc2.excp': (r2 >> 24) & 0x7f,
        'rsrc3.accum_offset': r3 & 0x3f, 'rsrc3.tg_split': (r3 >> 16) & 1,
        'props': hex(props), 'kernarg_preload': (w[14] >> 16) & 0xffff,
    }


def main():
    a = fields(kd_bytes(sys.argv[1]))
    b = fields(kd_bytes(sys.argv[2])) if len(sys.argv) > 2 else None
    for k, v in a.items():
        if b is None:
            print(f'{k:28s} {v}')
        else:
            mark = '' if b[k] == v else '   <-- differs'
            print(f'{k:28s} {str(v):>10s} {str(b[k]):>10s}{mark}')


if __name__ == '__main__':
    main()
