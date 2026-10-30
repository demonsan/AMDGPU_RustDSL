// (1) `forge_for_tests` may only exist under cfg(test)/feature test-support; the release of `verify_insts` is the public
// rewrite-checking hook but it cannot manufacture an AsmBlock.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    s.output(x);
    let blk = s.finish().unwrap();
    let bad = vec![AInst { op: "SNop", dst: None, srcs: vec![], imm: 99 }];
    let r = verify_insts(&blk, bad);
    println!("CASE pass_verify_insts: verify_insts on a malformed rewrite = {:?}", r.err());
    println!("  (verify_insts returns only Result<(), Vec<String>>; it cannot hand out an AsmBlock)");
}
