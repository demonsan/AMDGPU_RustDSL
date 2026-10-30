// ATTACK (f): second compare while vcc is live -> type error expected; and a second `vcc()` after the token
// was dropped is a runtime panic (not a compile error) - show both.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    let y = s.splat(2);
    let v = s.vcc();
    let live = s.v_cmp_lt_i32(v, &x, &y);
    let _ = live;                       // drop the Live token; vcc_taken stays true
    let v2 = s.vcc();                   // compiles, panics at runtime ("vcc already taken")
    let _ = v2;
    println!("CASE pass_second_vcc: compiled (the second vcc() is a runtime panic, not a compile error)");
}
