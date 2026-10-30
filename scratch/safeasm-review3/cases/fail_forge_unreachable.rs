// (F12) `forge_for_tests` must be unreachable in a normal (non-test, no `test-support` feature) build.
use amdgpu_safeasm::*;
fn main() {
    let _f = amdgpu_safeasm::forge_for_tests;
}
