import subprocess
p = 'crates/amdgpu-kernels/src/probes/g1250_mla_roles.rs'
t = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:90]); return s.replace(a, b)


t = rep(t, 'if mask > 63 {', 'if mask > 127 {')
t = rep(t, 'let s_chain = |b: &mut FnBuilder, kfr: &[Vec<ValueId>]| -> Vec<Opnd> {', 'let s_chain = |b: &mut FnBuilder, kfr: &[Vec<Opnd>]| -> Vec<Opnd> {')
t = rep(t, '    let s0 = s_chain(&mut b, &kf0);', '    let kf0o: Vec<Vec<Opnd>> = kf0.iter().map(|v| v.iter().map(|&x| Opnd::Val(x)).collect()).collect();\n    let s0 = s_chain(&mut b, &kf0o);')
# carried K vars (mask 4)
t = rep(t, '''    inits.push((DType::F32, Opnd::Val(neg)));
    inits.push((DType::F32, Opnd::Val(zf)));
    let st = b.declare_vars(&inits);''', '''    inits.push((DType::F32, Opnd::Val(neg)));
    inits.push((DType::F32, Opnd::Val(zf)));
    let pre = mask & 4 != 0;
    if pre {
        for _ in 0..(2 * 9) {
            let z = b.frag_cat(&[Opnd::I32(0); 8]);
            inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
        }
    }
    let st = b.declare_vars(&inits);
    let kvars: Vec<Vec<_>> = if pre { (0..2usize).map(|pg| (0..9usize).map(|c| st[10 + nx + pg * 9 + c]).collect()).collect() } else { Vec::new() };''')
# prologue: initial K fragments after the S0 stores
t = rep(t, '''    for k in w_n..3 * w_n {
        sbuf_store(&mut b, gc[k as usize], slot_c(k), &s0);
    }
    b.end_if();''', '''    for k in w_n..3 * w_n {
        sbuf_store(&mut b, gc[k as usize], slot_c(k), &s0);
    }
    if pre {
        let kf = k_frags(&mut b);
        for pg in 0..2usize {
            for c in 0..9usize {
                b.set_var(kvars[pg][c], Opnd::Val(kf[pg][c]));
            }
        }
    }
    b.end_if();''')
# the loop: K fragments from the variables
t = rep(t, '''                let kfr = if hoist { kfr_h.clone() } else { k_frags(b) };''', '''                let kfr: Vec<Vec<Opnd>> = if hoist {
                    kfr_h.iter().map(|v| v.iter().map(|&x| Opnd::Val(x)).collect()).collect()
                } else if pre {
                    kvars.iter().map(|v| v.iter().map(|&x| Opnd::Var(x)).collect()).collect()
                } else {
                    k_frags(b).iter().map(|v| v.iter().map(|&x| Opnd::Val(x)).collect()).collect()
                };''')
# end of the MMA loop: next window's K fragments before the barrier, then only the writes must complete
old = '''        b.wait_lds(0);
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nitr);
        b.break_if(fin);
        b.end_loop();
    };
'''
new = '''        if pre && do_s {
            // the next window's K fragments are issued before the barrier (their slot is already published); only the stores before them must complete
            let nk = k_frags(b);
            for pg in 0..2usize {
                for c in 0..9usize {
                    b.set_var(kvars[pg][c], Opnd::Val(nk[pg][c]));
                }
            }
            b.wait_lds_writes();
        } else {
            b.wait_lds(0);
        }
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nitr);
        b.break_if(fin);
        b.end_loop();
    };
'''
t = rep(t, old, new)
open(p, 'w', encoding='utf-8', newline='').write(t)
print('ok')
