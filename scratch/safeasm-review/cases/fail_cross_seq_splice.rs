// (i) there is deliberately no API to splice two sequences / loops / DS reads: prove it does not exist.
use amdgpu_safeasm::*;
fn main() {
    let mut s1 = Seq::new();
    let mut s2 = Seq::new();
    s1.splice(&mut s2);                      // expected: no method `splice` (cross-Seq contract not modelled)
}
