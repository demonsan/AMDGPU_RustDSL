//! Per-region scheduler strategies keyed by region NAME: the kernel declares them (`declare_region`), lowering stamps the backend's region ids (`weave_region_begin` / `weave_region_end`),
//! `region_strategies` maps every lowered region id to its strategy (the nearest declared region on the way up the nesting, 0 = the kernel default).
use amdgpu_codegen::{lower, region_strategies, scheduler_options};
use amdgpu_dsl::{FnBuilder, Function};

fn build(declare: bool) -> Function {
    let mut b = FnBuilder::new("w", 64);
    if declare {
        b.declare_region("hot", true, "ldsink=1");
        b.declare_region("other", false, "reorder");
    }
    b.barrier(); // no region
    b.weave_region("hot", |b| {
        b.barrier();
        // an undeclared region inherits its parent's strategy
        b.weave_region("plain", |b| b.barrier());
    });
    b.barrier();
    b.weave_region("other", |b| b.barrier());
    b.weave_region("hot", |b| b.barrier()); // a second INSTANCE of the same declared region: its own lowered id, the same strategy
    b.weave_region("undeclared", |b| b.barrier()); // undeclared at the top level: the kernel default
    b.finish()
}

fn strategies(f: &Function) -> Vec<u16> {
    let m = lower::lower(f).expect("lowers");
    let map = region_strategies(f, &m);
    m.blocks.iter().flat_map(|b| b.insts.iter().map(|i| map[i.region as usize])).collect()
}

#[test]
fn declared_regions_map_to_strategies_by_name_with_inheritance() {
    let f = build(true);
    // barriers in program order: outside, hot, plain (inside hot), outside, other, hot (2nd instance), undeclared
    assert_eq!(strategies(&f), vec![0, 1, 1, 0, 2, 1, 0]);
}

#[test]
fn without_declarations_every_instruction_keeps_the_default() {
    let f = build(false);
    assert!(strategies(&f).iter().all(|&x| x == 0));
    assert!(scheduler_options(&f).map_or(true, |o| o.regions.is_empty()), "no declarations: the scheduler options carry no region table");
}

#[test]
fn the_declared_regions_reach_the_scheduler_options() {
    let f = build(true);
    let o = scheduler_options(&f).expect("regions turn the scheduler on");
    assert_eq!(o.regions.len(), 2);
    assert!(o.regions[0].authored && o.regions[0].name == "hot" && o.regions[0].tokens == "ldsink=1");
    assert!(!o.regions[1].authored && o.regions[1].name == "other");
    assert!(!o.reorder, "a kernel without a preset: the default is rewait (waits only)");
}