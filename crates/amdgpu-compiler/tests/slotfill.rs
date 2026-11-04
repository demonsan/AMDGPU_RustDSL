//! `slotfill` (sched/slotfill.rs): MFMA skeleton kept, floaters spread over their deadline windows, dependences / fences / the DMA <-> ds_read contract honoured.

mod common;

use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::order::RELAX_DMA_LDS_READ;
use amdgpu_compiler::sched::{slotfill, verify, Options};
use common::func;
use std::sync::atomic::Ordering;

// registers: 0..16 = accumulators (V4), 16 = A frag (V8), 17 = B frag (V8), 18 = scale (V1), 19 = lds address (V1), 20.. = ds_read destinations (V4) x 24,
// 44.. = V1 scratch x 8, 52 = rsrc (S4), 53 = soffset (S1), 54 = m0 source (S1), 55 = voffset (V1)
fn mk(insts: Vec<MInst>) -> MFunction {
    let mut v = vec![RClass::V4; 16];
    v.push(RClass::V8);
    v.push(RClass::V8);
    v.push(RClass::V1);
    v.push(RClass::V1);
    for _ in 0..24 {
        v.push(RClass::V4);
    }
    for _ in 0..8 {
        v.push(RClass::V1);
    }
    v.push(RClass::S4);
    v.push(RClass::S1);
    v.push(RClass::S1);
    v.push(RClass::V1);
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], v)
}

fn mfma(k: u32) -> MInst {
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(k).srcreg(16).srcreg(17).srcreg(k).srcreg(18).srcreg(18)
}

fn ds_read(d: u32) -> MInst {
    MInst::new(MOp::LdsReadB128).def(20 + d).srcreg(19).imm((d * 16) as i64)
}

fn dma(k: i64) -> MInst {
    MInst::new(MOp::BufferLoadDwordX4Lds).srcreg(52).srcreg(55).srcreg(53).srcreg(54).imm(k * 1024)
}

fn name(i: &MInst) -> String {
    match i.op {
        MOp::VMfmaScale16x16x128Fp8 => format!("M{}", i.dst.unwrap()),
        MOp::LdsReadB128 => format!("R{}", i.dst.unwrap() - 20),
        MOp::BufferLoadDwordX4Lds => format!("D{}", i.imm / 1024),
        MOp::SBarrier => "B".into(),
        MOp::VAddU32 => "a".into(),
        MOp::VMovB32 => "m".into(),
        other => format!("{other:?}"),
    }
}

fn run(insts: Vec<MInst>) -> Vec<String> {
    let mut m = mk(insts);
    let o = Options { verify: true, ..Options::default() };
    slotfill::run(&mut m, &o).expect("slotfill + verifier");
    m.blocks[0].insts.iter().map(name).collect()
}

fn mfmas(n: u32) -> Vec<MInst> {
    (0..n).map(mfma).collect()
}

fn pos(v: &[String], s: &str) -> usize {
    v.iter().position(|x| x == s).unwrap_or_else(|| panic!("{s} missing in {v:?}"))
}

#[test]
fn the_mfma_order_is_the_skeleton_and_nothing_is_lost() {
    let mut insts: Vec<MInst> = (0..8).map(ds_read).collect();
    insts.extend(mfmas(16));
    let out = run(insts);
    assert_eq!(out.len(), 24);
    let ms: Vec<&String> = out.iter().filter(|s| s.starts_with('M')).collect();
    assert_eq!(ms, (0..16).map(|k| format!("M{k}")).collect::<Vec<_>>().iter().collect::<Vec<_>>());
    for d in 0..8 {
        assert!(out.contains(&format!("R{d}")));
    }
}

#[test]
fn a_burst_of_independent_loads_is_spread_over_the_mfma_stream() {
    let mut insts: Vec<MInst> = (0..8).map(ds_read).collect();
    insts.extend(mfmas(16));
    let out = run(insts);
    // source: R0..R7 then M0..M15 (a burst in front of the first MFMA). Now MFMAs interleave with the reads: the last read comes after at least 6 MFMAs
    assert!(pos(&out, "R7") > pos(&out, "M5"), "{out:?}");
    // and the reads keep a gap: no more than 2 reads in a row
    let mut run_len = 0;
    for s in &out {
        if s.starts_with('R') {
            run_len += 1;
            assert!(run_len <= 3, "{out:?}");
        } else {
            run_len = 0;
        }
    }
}

#[test]
fn a_producer_chain_travels_with_its_load_and_stays_before_it() {
    // m = movb; a = add(m, m); ds_read at address `a`... the chain uses scratch regs 44..46, the read uses 45 as its address
    let chain = vec![
        MInst::new(MOp::VMovB32).def(44).imm(7),
        MInst::new(MOp::VAddU32).def(45).srcreg(44).srcreg(44),
        MInst::new(MOp::LdsReadB128).def(20).srcreg(45).imm(0),
    ];
    let mut insts = chain;
    insts.extend(mfmas(16));
    let out = run(insts);
    assert!(pos(&out, "m") < pos(&out, "a") && pos(&out, "a") < pos(&out, "R0"), "{out:?}");
}

#[test]
fn a_load_feeding_a_late_mfma_meets_its_deadline_and_a_reload_waits_for_the_last_reader() {
    // R0 writes the A fragment (reg 16) that every MFMA reads: it must stay in front of M0 (RAW); a second read writes reg 20+1 only
    let mut insts = vec![MInst::new(MOp::LdsReadB128).def(16).srcreg(19).imm(0)];
    insts.extend(mfmas(16));
    // a reload of the A fragment after the stream (WAR with every MFMA): must stay behind M15
    insts.push(MInst::new(MOp::LdsReadB128).def(16).srcreg(19).imm(64));
    let out = run(insts);
    let first_read = out.iter().position(|s| s.starts_with('R')).unwrap();
    let last_read = out.iter().rposition(|s| s.starts_with('R')).unwrap();
    assert!(first_read < pos(&out, "M0"), "{out:?}");
    assert!(last_read > pos(&out, "M15"), "{out:?}");
}

#[test]
fn nothing_crosses_a_barrier() {
    let mut insts: Vec<MInst> = (0..4).map(ds_read).collect();
    insts.extend(mfmas(10));
    insts.push(MInst::new(MOp::SBarrier));
    insts.extend((10..16).map(mfma));
    insts.extend((4..8).map(ds_read));
    let out = run(insts);
    let b = pos(&out, "B");
    for d in 0..4 {
        assert!(pos(&out, &format!("R{d}")) < b);
    }
    for d in 4..8 {
        assert!(pos(&out, &format!("R{d}")) > b);
    }
    assert_eq!(out.iter().take(b).filter(|s| s.starts_with('M')).count(), 10);
}

#[test]
fn dma_issues_spread_among_ds_reads_only_under_the_contract() {
    // 8 DMAs (in order) then 8 ds_reads then 16 MFMAs: with the relaxation the DMAs interleave with MFMAs and may pass ds_reads
    let mut insts: Vec<MInst> = (0..8).map(|k| dma(k)).collect();
    insts.extend((0..8).map(ds_read));
    insts.extend(mfmas(16));
    let out = run(insts);
    let d: Vec<usize> = (0..8).map(|k| pos(&out, &format!("D{k}"))).collect();
    assert!(d.windows(2).all(|w| w[0] < w[1]), "DMAs keep their relative order: {out:?}");
    assert!(*d.last().unwrap() > pos(&out, "M4"), "{out:?}");
    // outside the pass the strict rules are back: a DMA and a ds_read may not swap
    assert!(!RELAX_DMA_LDS_READ.load(Ordering::Relaxed));
    let strict = mk(vec![dma(0), ds_read(0)]);
    let i = strict.blocks[0].insts.clone();
    assert!(verify::check_perm(&strict, &i, &[1, 0], &Term::Endpgm, true).is_err());
}

#[test]
fn a_block_with_too_few_mfmas_is_left_alone() {
    let mut insts: Vec<MInst> = (0..4).map(ds_read).collect();
    insts.extend(mfmas(6));
    let before: Vec<String> = insts.iter().map(name).collect();
    assert_eq!(run(insts), before);
}

#[test]
fn explicit_partial_waits_make_the_block_untouched() {
    let mut insts: Vec<MInst> = (0..4).map(ds_read).collect();
    insts.push(MInst::new(MOp::SWaitLgkm).imm(2));
    insts.extend(mfmas(16));
    let before: Vec<String> = insts.iter().map(name).collect();
    assert_eq!(run(insts), before);
}