// error-contains: (pass case)
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let sb = s.splat(1);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let o = t.retire(&mut s);
    let lane = s.lane::<16, 3>(&o);
    let mut x = s.vgpr::<1>();
    s.v_add_f32(&mut x, lane, &sa);
    let vcc = s.vcc().unwrap();
    let live = s.v_cmp_lt_i32(vcc, Imm::<5>::new(), &x);
    let mut d = s.vgpr::<1>();
    let _free = s.v_cndmask_b32(live, &mut d, &sa, &sb);
    let mut sg = s.sgpr::<1>();
    let r = s.ready(&d);
    s.v_readfirstlane_b32(&mut sg, r);
    // loads: issue, then wait (partial count derived from the scoreboard)
    let addr = s.splat(0);
    let (d0, d1) = (s.vgpr::<4>(), s.vgpr::<4>());
    let r0 = s.ds_read_b128::<0>(d0, &addr);
    let r1 = s.ds_read_b128::<16>(d1, &addr);
    let v0 = r0.wait(&mut s);
    let v1 = r1.wait(&mut s);
    s.output(v0);
    s.output(v1);
    let _ = s.finish();
}