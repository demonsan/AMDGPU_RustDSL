import re


def edit(path, fn):
    raw = open(path, encoding='utf-8', newline='').read(); crlf = '\r\n' in raw
    t = fn(raw.replace('\r\n', '\n'))
    open(path, 'w', encoding='utf-8', newline='').write(t.replace('\n', '\r\n') if crlf else t)


def kern(t):
    a = 'pub fn gemm_tdm_stage_f4_256x256_ks2_epit_swz('
    i = t.index(a)
    j = t.index('\n}\n', i) + 3
    add = '''
/// The best 256 x 256 kernel on a persistent grid (host PERSIST=G): no cross-tile overlap (the LDS-staged epilogue uses the whole LDS), only the tile loop.
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_persist(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, persistent: true, ..RingOpts::default() })
}
'''
    return t[:j] + add + t[j:]


edit('crates/amdgpu-kernels/src/probes/g1250_gemm.rs', kern)


def reg(t):
    k = '"gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles"'
    p = t.index(k)
    t = t[:p] + '"gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax_persist", ' + t[p:]
    arm = re.search(r'(?m)^\s+"gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles" =>.*$', t).group(0)
    return t.replace(arm, arm + '''
        "gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax_persist" => Some(crate::probes::g1250_gemm::gemm_tdm_stage_f4_256x256_ks2_epit_persist(name, true, true)),''')


edit('crates/amdgpu-kernels/src/registry.rs', reg)
print('ok')
