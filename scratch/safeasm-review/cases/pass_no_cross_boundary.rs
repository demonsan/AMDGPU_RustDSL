// ATTACK (i)/(j)/(k): cross-Seq / loop / DS-VMEM hazards are not expressible in the safe API (no bridge, no
// loops, no DS/VMEM ops). This case documents that the only way is the raw-MIR escape hatch.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut w = s.splat(0);                 // VALU write
    let a = s.zeroed::<8>();
    let sa = s.splat(127);
    let mut o = s.vgpr::<16>();
    // no way to splice a second Seq / a loop / a DS read into `s`: the only "boundary" is to_mfunction()
    let m = s.to_mfunction("no_cross_boundary");
    println!("CASE pass_no_cross_boundary: compiled; Seq has no splice/loop/DS ops (see pass_escape_mfunction for the raw path)");
    let _ = (&mut w, &a, &m, &o, &sa);
}
