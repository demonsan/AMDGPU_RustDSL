// (1f) Exit has no public constructor: a matrix result cannot be turned into plain data and reused. Expected: private fields.
use amdgpu_safeasm::*;
fn main() {
    let e = Exit::<16> { id: 0, issued: 0 };
    let _ = e;
}
