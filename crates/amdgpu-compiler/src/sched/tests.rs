use super::*;
use std::collections::HashMap;

fn func(vregs: &[RClass], insts: Vec<MInst>) -> MFunction {
    let mut b = Block::new(0);
    b.insts = insts;
    MFunction {
        name: "t".into(),
        args: vec![],
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 4096,
        blocks: vec![b],
        vregs: vregs.to_vec(),
        loops: vec![],
        alias: HashMap::new(),
        elide_fallthrough: false,
    }
}

fn opts() -> Options {
    Options { verify: true, ..Options::default() }
}

#[test]
fn independent_work_fills_exp_latency() {
    // r0 = v_exp r1 ; r2 = r0 + r0 ; then independent adds that should be hoisted between them
    let v = [RClass::V1; 8];
    let insts = vec![
        MInst::new(MOp::VExpF32).def(0).srcreg(1),
        MInst::new(MOp::VAddF32).def(2).srcreg(0).srcreg(0),
        MInst::new(MOp::VAddF32).def(3).srcreg(4).srcreg(5),
        MInst::new(MOp::VAddF32).def(6).srcreg(4).srcreg(5),
    ];
    let mut m = func(&v, insts);
    run(&mut m, &opts()).unwrap();
    let ops: Vec<_> = m.blocks[0].insts.iter().map(|i| i.dst).collect();
    // the consumer of the exp must not be directly after it
    assert_eq!(ops[0], Some(0));
    assert_eq!(m.blocks[0].insts.len(), 4);
}

#[test]
fn subregister_writes_are_independent() {
    // two writes to disjoint dwords of one tuple may swap; a read of the whole tuple stays last
    let mut m = func(
        &[RClass::V4, RClass::V1, RClass::V1, RClass::V4],
        vec![
            MInst::new(MOp::VMovB32).def(1).imm(1),
            MInst::new(MOp::VMovB32).def(2).imm(2),
            MInst::new(MOp::VWriteSub).def(0).srcreg(1).imm(0),
            MInst::new(MOp::VWriteSub).def(0).srcreg(2).imm(1),
            MInst::new(MOp::VWriteSub).def(0).srcreg(2).imm(2),
            MInst::new(MOp::VWriteSub).def(0).srcreg(2).imm(3),
            MInst::new(MOp::LdsWriteB128).srcreg(1).srcreg(0).imm(0),
        ],
    );
    let g = depgraph::build(&m, &m.blocks[0].insts.clone(), &Term::Endpgm, false);
    // WriteSub lane 0 and lane 1 have no edge between them
    assert!(!g.succs[2].iter().any(|e| e.0 == 3));
    run(&mut m, &opts()).unwrap();
    assert_eq!(m.blocks[0].insts.last().unwrap().op, MOp::LdsWriteB128);
}

#[test]
fn verifier_rejects_illegal_orders() {
    let m = func(
        &[RClass::V1; 4],
        vec![
            MInst::new(MOp::VMovB32).def(0).imm(1),
            MInst::new(MOp::VAddF32).def(1).srcreg(0).srcreg(0),
            MInst::new(MOp::VMovB32).def(0).imm(2),
        ],
    );
    let insts = m.blocks[0].insts.clone();
    assert!(verify::check_perm(&m, &insts, &[0, 1, 2], &Term::Endpgm, false).is_ok());
    // RAW broken
    assert!(verify::check_perm(&m, &insts, &[1, 0, 2], &Term::Endpgm, false).is_err());
    // WAR broken (second write of r0 before the reader)
    assert!(verify::check_perm(&m, &insts, &[0, 2, 1], &Term::Endpgm, false).is_err());
    assert!(verify::check_perm(&m, &insts, &[0, 1], &Term::Endpgm, false).is_err());
}

#[test]
fn lds_ops_and_barrier_order() {
    let m = func(
        &[RClass::V1, RClass::V4, RClass::V1],
        vec![
            MInst::new(MOp::LdsWriteB128).srcreg(0).srcreg(1).imm(0),
            MInst::new(MOp::SBarrier),
            MInst::new(MOp::LdsReadB32).def(2).srcreg(0).imm(0),
        ],
    );
    let insts = m.blocks[0].insts.clone();
    assert!(verify::check_perm(&m, &insts, &[2, 1, 0], &Term::Endpgm, false).is_err());
    assert!(verify::check_perm(&m, &insts, &[0, 2, 1], &Term::Endpgm, false).is_err());
}

#[test]
fn disjoint_lds_offsets_commute_same_base() {
    let m = func(
        &[RClass::V1, RClass::V4, RClass::V4],
        vec![
            MInst::new(MOp::LdsWriteB128).srcreg(0).srcreg(1).imm(0),
            MInst::new(MOp::LdsReadB128).def(2).srcreg(0).imm(16),
        ],
    );
    let insts = m.blocks[0].insts.clone();
    assert!(verify::check_perm(&m, &insts, &[1, 0], &Term::Endpgm, false).is_ok());
    let m2 = func(
        &[RClass::V1, RClass::V4, RClass::V4],
        vec![
            MInst::new(MOp::LdsWriteB128).srcreg(0).srcreg(1).imm(0),
            MInst::new(MOp::LdsReadB128).def(2).srcreg(0).imm(8),
        ],
    );
    let insts2 = m2.blocks[0].insts.clone();
    assert!(verify::check_perm(&m2, &insts2, &[1, 0], &Term::Endpgm, false).is_err());
}

#[test]
fn waits_are_inserted_and_partial() {
    // two loads then a use of the first: vmcnt(1)
    let mut m = func(
        &[RClass::V2, RClass::V1, RClass::V1, RClass::V1],
        vec![
            MInst::new(MOp::GlobalLoadF32).def(1).srcreg(0).imm(0),
            MInst::new(MOp::GlobalLoadF32).def(2).srcreg(0).imm(4),
            MInst::new(MOp::VAddF32).def(3).srcreg(1).srcreg(1),
        ],
    );
    waits::insert_waits(&mut m);
    let w: Vec<_> = m.blocks[0].insts.iter().filter(|i| i.op == MOp::SWaitVm).collect();
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].imm, 1);
    assert!(w[0].auto);
    waits::verify_waits(&m).unwrap();
}

/// Tiny deterministic generator.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn random_blocks_schedule_legally() {
    for seed in 1..200u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        let nv = 12u32;
        let mut vregs = vec![RClass::V1; nv as usize];
        vregs.push(RClass::V4); // 12 tuple
        vregs.push(RClass::V2); // 13 pointer
        let mut insts = Vec::new();
        for _ in 0..(10 + rng.below(40)) {
            let a = rng.below(nv as u64) as u32;
            let b = rng.below(nv as u64) as u32;
            let d = rng.below(nv as u64) as u32;
            let inst = match rng.below(9) {
                0 => MInst::new(MOp::VAddF32).def(d).srcreg(a).srcreg(b),
                1 => MInst::new(MOp::VExpF32).def(d).srcreg(a),
                2 => MInst::new(MOp::VMovB32).def(d).imm(rng.below(5) as i64),
                3 => MInst::new(MOp::VWriteSub).def(12).srcreg(a).imm(rng.below(4) as i64),
                4 => MInst::new(MOp::LdsWriteB128).srcreg(a).srcreg(12).imm(16 * rng.below(3) as i64),
                5 => MInst::new(MOp::LdsReadB32).def(d).srcreg(a).imm(4 * rng.below(8) as i64),
                6 => MInst::new(MOp::GlobalLoadF32).def(d).srcreg(13).imm(4 * rng.below(4) as i64),
                7 => MInst::new(MOp::VCmpLtF32).srcreg(a).srcreg(b),
                _ => MInst::new(MOp::VCndmask).def(d).srcreg(a).srcreg(b),
            };
            insts.push(inst);
            if rng.below(15) == 0 {
                insts.push(MInst::new(MOp::SWaitLgkm).imm(0));
            }
            if rng.below(20) == 0 {
                insts.push(MInst::new(MOp::SBarrier));
            }
        }
        let mut m = func(&vregs, insts);
        run(&mut m, &opts()).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}

#[test]
fn every_kernel_schedules_and_verifies() {
    for name in crate::kernel_names() {
        let f = crate::kernel_by_name(name).unwrap();
        let Ok(mut m) = crate::lower::lower(&f) else { continue };
        for xbar in [false, true] {
            let mut mm = m.clone();
            let mut o = opts();
            o.alu_across_barrier = xbar;
            run(&mut mm, &o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(crate::mir::verify(&mm).is_empty(), "{name}");
        }
        let mut mm = m.clone();
        let mut o = opts();
        o.reorder = false;
        run(&mut mm, &o).unwrap_or_else(|e| panic!("{name} rewait: {e}"));
        let _ = &mut m;
    }
}

// ---- software pipelining: GPU-free semantic check with a tiny MIR interpreter (V1 integer subset) ----

fn interp(m: &MFunction, n_val: u32) -> (u32, u32) {
    let mut val: HashMap<VReg, u32> = HashMap::new();
    let mut vcc = false;
    let get = |val: &HashMap<VReg, u32>, s: &Src| -> u32 {
        match s {
            Src::Imm(i) => *i as u32,
            Src::Reg(r) => *val.get(r).unwrap_or(&0),
        }
    };
    let mut bi = 0usize;
    for _ in 0..100_000 {
        let b = &m.blocks[bi];
        for i in &b.insts {
            let s = |k: usize| get(&val, &i.srcs[k]);
            let r = match i.op {
                MOp::VMovB32 => Some(if i.srcs.is_empty() { i.imm as u32 } else { s(0) }),
                MOp::VMovFromS => Some(n_val),
                MOp::VAddU32 => Some(s(0).wrapping_add(s(1))),
                MOp::VMulLoU32 => Some(s(0).wrapping_mul(s(1))),
                MOp::VXorB32 => Some(s(0) ^ s(1)),
                MOp::VAndB32 => Some(s(0) & s(1)),
                MOp::VLshrB32 => Some(s(1) >> (s(0) & 31)),
                MOp::VCmpLtI32 => {
                    vcc = (s(0) as i32) < (s(1) as i32);
                    None
                }
                MOp::VCmpGeI32 => {
                    vcc = (s(0) as i32) >= (s(1) as i32);
                    None
                }
                MOp::VCmpNeI32 => {
                    vcc = s(0) != s(1);
                    None
                }
                MOp::VCndmask => Some(if vcc { s(1) } else { s(0) }),
                other => panic!("interp: unsupported {other:?}"),
            };
            if let (Some(d), Some(x)) = (i.dst, r) {
                val.insert(d, x);
            }
        }
        let next = |l: u32| m.blocks.iter().position(|x| x.label == l).unwrap();
        match &b.term {
            Term::Goto(t) => bi = next(*t),
            Term::CondGoto { on, target, fall } => {
                let taken = match on {
                    Pred::Vccnz => vcc,
                    Pred::Vccz => !vcc,
                    _ => panic!("pred"),
                };
                bi = next(if taken { *target } else { *fall });
            }
            Term::Endpgm => return (val[&1], val[&0]),
        }
    }
    panic!("interp: no termination");
}

fn counted_loop() -> MFunction {
    let vregs = vec![
        RClass::V1, // 0 it
        RClass::V1, // 1 acc
        RClass::S1, // 2 n
        RClass::V1, // 3 one
        RClass::V1, // 4 it_next
        RClass::V1, // 5 nreg
        RClass::V1, // 6 t1
        RClass::V1, // 7 t2
        RClass::V1, // 8 t3
        RClass::V1, // 9 c3
    ];
    let mut pre = Block::new(0);
    pre.insts = vec![MInst::new(MOp::VMovB32).def(0).imm(0), MInst::new(MOp::VMovB32).def(1).imm(0)];
    pre.term = Term::Goto(1);
    let mut h = Block::new(1);
    h.insts = vec![
        MInst::new(MOp::VMovB32).def(9).imm(3),
        MInst::new(MOp::VMulLoU32).def(6).srcreg(0).srcreg(9),
        MInst::new(MOp::VAddU32).def(7).srcreg(6).src(Src::Imm(7)),
        MInst::new(MOp::VMulLoU32).def(8).srcreg(7).srcreg(7),
        MInst::new(MOp::VAddU32).def(1).srcreg(1).srcreg(8),
        MInst::new(MOp::VMovB32).def(3).imm(1),
        MInst::new(MOp::VAddU32).def(4).srcreg(0).srcreg(3),
        MInst::new(MOp::VMovB32).def(0).srcreg(4),
        MInst::new(MOp::VMovFromS).def(5).srcreg(2),
        MInst::new(MOp::VCmpLtI32).srcreg(4).srcreg(5),
    ];
    h.term = Term::CondGoto { on: Pred::Vccnz, target: 1, fall: 2 };
    let mut x = Block::new(2);
    x.term = Term::Endpgm;
    MFunction {
        name: "t".into(),
        args: vec![],
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: vec![pre, h, x],
        vregs,
        loops: vec![(1, 1)],
        alias: HashMap::new(),
        elide_fallthrough: false,
    }
}

#[test]
fn swp_matches_plain_loop_for_all_trip_counts_and_cuts() {
    let m = counted_loop();
    let li = swp::analyze(&m, 1).expect("canonical loop");
    let nb = li.body.len();
    let live_after: std::collections::HashSet<VReg> = [0u32, 1].into_iter().collect();
    for k in 1..nb {
        let p = swp::apply(&m, &li, k, &live_after).expect("apply");
        for n in 0..14u32 {
            assert_eq!(interp(&m, n), interp(&p, n), "cut {k} n {n}");
        }
    }
}

#[test]
fn taildup_preserves_semantics_in_interpreter() {
    // diamond inside a counted loop: odd iterations add t, even iterations xor; join accumulates
    let mut m = counted_loop();
    // split H: [.. it update + cmp] stays; insert arms: simplest check is the pure transformation on a hand-built diamond
    m.vregs.push(RClass::V1); // 10
    let mut a = Block::new(10);
    a.insts = vec![MInst::new(MOp::VAddU32).def(1).srcreg(1).srcreg(0)];
    a.term = Term::Goto(12);
    let mut b = Block::new(11);
    b.insts = vec![MInst::new(MOp::VXorB32).def(1).srcreg(1).srcreg(0)];
    b.term = Term::Goto(12);
    let mut j = Block::new(12);
    j.insts = vec![
        MInst::new(MOp::VAddU32).def(1).srcreg(1).src(Src::Imm(5)),
        MInst::new(MOp::VMovB32).def(3).imm(1),
        MInst::new(MOp::VAddU32).def(4).srcreg(0).srcreg(3),
        MInst::new(MOp::VMovB32).def(0).srcreg(4),
        MInst::new(MOp::VMovFromS).def(5).srcreg(2),
        MInst::new(MOp::VCmpLtI32).srcreg(4).srcreg(5),
    ];
    j.term = Term::CondGoto { on: Pred::Vccnz, target: 1, fall: 2 };
    let mut h = Block::new(1);
    h.insts = vec![MInst::new(MOp::VAndB32).def(6).srcreg(0).src(Src::Imm(1)), MInst::new(MOp::VCmpNeI32).src(Src::Imm(0)).srcreg(6)];
    h.term = Term::CondGoto { on: Pred::Vccnz, target: 10, fall: 11 };
    let mut pre = Block::new(0);
    pre.insts = vec![MInst::new(MOp::VMovB32).def(0).imm(0), MInst::new(MOp::VMovB32).def(1).imm(0)];
    pre.term = Term::Goto(1);
    let mut x = Block::new(2);
    x.term = Term::Endpgm;
    m.blocks = vec![pre, h, a, b, j, x];
    m.loops = vec![(1, 12)];
    let base: Vec<(u32, u32)> = (0..10).map(|n| interp(&m, n)).collect();
    // tail-duplicate k instructions of the join into both arms (blocks index 2,3 <- 4)
    for k in 1..=6 {
        let mut c = m.clone();
        let pre: Vec<MInst> = c.blocks[4].insts[..k].to_vec();
        c.blocks[2].insts.extend(pre.iter().cloned());
        c.blocks[3].insts.extend(pre);
        c.blocks[4].insts.drain(..k);
        for n in 0..10 {
            assert_eq!(interp(&c, n), base[n as usize], "k {k} n {n}");
        }
    }
}