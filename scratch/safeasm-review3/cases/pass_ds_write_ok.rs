// (F10) ds_write must be usable now: it takes a scoreboard slot without a token, wait_lgkm_all covers it.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let mut data = s.zeroed::<1>();
    s.ds_write_b32(&addr, &data, 0);
    s.wait_lgkm_all();
    let z = s.splat(99);
    s.v_mov_b32(&mut data, &z);
    s.output(data);
    match s.finish() {
        Ok(b) => println!("CASE pass_ds_write_ok: finish Ok: {:?}", b.insts().iter().map(|i| i.op).collect::<Vec<_>>()),
        Err(e) => println!("CASE pass_ds_write_ok: finish Err: {e}"),
    }
}
