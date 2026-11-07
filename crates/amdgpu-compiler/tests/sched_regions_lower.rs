//! Weave region scopes (`FnBuilder::weave`, `Node::Weave`): lowering stamps the declared region's id (`MInst::region`) on the instructions emitted inside the scope; the scope itself emits no code.
use amdgpu_compiler::{lower, policy_name, scheduler_options};
use amdgpu_dsl::{FnBuilder, Function, Node};

fn regions_of(f: &Function) -> Vec<u16> {
    let m = lower::lower(f).expect("lowers");
    m.blocks.iter().flat_map(|b| b.insts.iter().map(|i| i.region)).collect()
}

fn build(declare: bool) -> Function {
    let mut b = FnBuilder::new("w", 64);
    if declare {
        b.declare_region("hot", true, "ldsink=1");
    }
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let x = b.add_i32(c3, c4); // region 0
    let y = b.weave("hot", |b| {
        let t = b.add_i32(x, c3);
        // an undeclared scope is transparent: its instructions stay in the enclosing region
        b.weave("plain", |b| b.add_i32(t, c4))
    });
    let _z = b.add_i32(y, c3); // region 0 again
    b.finish()
}

#[test]
fn the_declared_region_is_stamped_on_its_scope_and_restored_after_it() {
    let f = build(true);
    assert!(f.body().iter().any(|n| matches!(n, Node::Weave { name, .. } if name == "hot")));
    let r = regions_of(&f);
    assert!(r.iter().any(|&x| x == 1), "something is lowered inside the scope: {r:?}");
    assert!(r.iter().any(|&x| x == 0), "and something outside");
    // 0..0 1..1 0..0: one contiguous run of the region
    let first = r.iter().position(|&x| x == 1).unwrap();
    let last = r.iter().rposition(|&x| x == 1).unwrap();
    assert!(r[first..=last].iter().all(|&x| x == 1), "one contiguous run: {r:?}");
    assert!(r[..first].iter().all(|&x| x == 0) && r[last + 1..].iter().all(|&x| x == 0), "nothing else is stamped: {r:?}");
}

#[test]
fn without_a_declaration_the_scope_stamps_nothing() {
    let f = build(false);
    assert!(regions_of(&f).iter().all(|&x| x == 0), "an undeclared name is the kernel default");
    assert!(scheduler_options(&f, policy_name(&f)).is_none() || scheduler_options(&f, policy_name(&f)).unwrap().regions.is_empty());
}

#[test]
fn the_declared_regions_reach_the_scheduler_options() {
    let f = build(true);
    let o = scheduler_options(&f, policy_name(&f)).expect("regions turn the scheduler on");
    assert_eq!(o.regions.len(), 1);
    assert!(o.regions[0].authored && o.regions[0].name == "hot" && o.regions[0].tokens == "ldsink=1");
    assert!(!o.reorder, "a kernel without a preset: the default is rewait (waits only)");
    // the flat form round-trips (weave markers are structure, not code)
    assert!(Function::parse_flat(f.stmts().to_vec()).is_ok());
}