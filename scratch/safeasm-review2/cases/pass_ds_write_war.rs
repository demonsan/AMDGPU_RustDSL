// (NEW, safe path) DS write data operand: `ds_write_b32` releases its data borrow immediately, so the register can be
// overwritten on the next instruction. The verifier only models matrix WAR, not an LSU data-operand WAR.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let mut data = s.zeroed::<1>();
    s.ds_write_b32(&addr, &data, 0);
    let z = s.splat(99);
    s.v_mov_b32(&mut data, &z);          // WAR on the ds_write data operand, one instruction later
    s.output(data);
    match s.finish() {
        Ok(b) => println!("CASE pass_ds_write_war: finish() accepted an immediate overwrite of a ds_write data operand: {:?}", b.insts().iter().map(|i| i.op).collect::<Vec<_>>()),
        Err(e) => println!("CASE pass_ds_write_war: refused: {e}"),
    }
}
