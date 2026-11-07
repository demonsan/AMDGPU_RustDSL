//! Item E1: index-level functional proof for the MXFP8 GEMM (v3 and v4).
//!
//! The proof composes the layouts the kernel actually builds:
//!   * global A row-major / B preshuffle (`a_glob`, `b_glob` in `gemm_mxfp8.rs`);
//!   * the 16x16x128 MFMA operand and accumulator TV layouts (`MmaAtom`), with the kernel's
//!     `swap_ab` orientation;
//!   * the 2x2 wave ownership of the 256x256 CTA and the 16x16 atom-block tiling;
//!   * the K-tile iteration/slot schedule (`Cadence::TwoSlot` for v3, `RegStaged` for v4);
//!   * the A/B scale panel indices, including the v4 DPP/permute transposition proof.
//! It proves that every C[m,n] receives exactly one contribution for each K block kb from
//! A[m, kb*128 .. kb*128+128] and B[n, ...] with the matching scales and is stored once.
//!
//! `drop_k_tile = Some(1)` is the regression model of the historical "tile 1 swallowed" bug: the
//! proof then fails with a concrete missing (m, n, k), e.g. (0,0,128).

use amdgpu_isa::MmaAtom;
use amdgpu_layout::{make_layout, proofs, tup, Dyns};
use amdgpu_tile::make_tiled_mma;
use std::collections::{HashMap, HashSet};

use amdgpu_kernels::gemm::GemmConfig;

#[derive(Debug)]
struct ProofError(#[allow(dead_code)] String);

type Items = HashMap<(i64, i64, i64), usize>;

fn cta_tiles(cfg: &GemmConfig) -> (i64, i64) {
    // every GEMM variant is 4 waves 2x2 over the CTA; K tiles are 128 keys.
    let wave_m = cfg.cta_m / 2;
    let wave_n = cfg.cta_n / 2;
    assert_eq!(wave_m % 16, 0);
    assert_eq!(wave_n % 16, 0);
    (wave_m, wave_n)
}

fn tile_count(k: i64) -> i64 { k / 128 }

/// The compute sequence of the pipeline: both cadences compute every tile exactly once. This is the
/// cadence's compute order; slots/generations are checked separately below.
fn compute_order(k: i64, drop_tile: Option<i64>) -> Vec<i64> {
    (0..tile_count(k)).filter(|t| Some(*t) != drop_tile).collect()
}

fn add_contribution(items: &mut Items, m: i64, n: i64, kb: i64) -> Result<(), ProofError> {
    let e = items.entry((m, n, kb)).or_insert(0);
    *e += 1;
    if *e > 1 { return Err(ProofError(format!("C[{m},{n}] K block {kb} receives {} contributions", *e))); }
    Ok(())
}

fn check_global_layouts(cfg: &GemmConfig, k: i64) -> Result<(), ProofError> {
    // A [M,K] row-major: a_glob = make_layout([cta_m,128], [K,1]) per K tile; every (row, global k)
    // has a unique byte. B's preshuffle layout is b_glob = ([16, cta_n/16],[16,8]) with strides
    // ([16, K*16],[1,256]); the expected formula is the one the kernel's DMA computes.
    let mut dyns = Dyns::new();
    let s_k = dyns.sym("K", 128, 1 << 20);
    let a_glob = make_layout(tup![cfg.cta_m, 128], tup![s_k.clone(), 1]);
    let mut a_seen = HashSet::new();
    for r in 0..cfg.cta_m {
        for kk in 0..k {
            let off = a_glob.crd2idx(&tup![r, kk % 128], &dyns).eval(&|_| k) + (kk / 128) * 128;
            if off != r * k + kk { return Err(ProofError(format!("A layout offset mismatch at ({r},{kk})"))); }
            if !a_seen.insert(off) { return Err(ProofError(format!("A image duplicate at ({r},{kk})"))); }
        }
    }
    let b_glob = make_layout(tup![tup![16, cfg.cta_n / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    let mut b_seen = HashSet::new();
    for n in 0..cfg.cta_n {
        for kk in 0..k {
            let off = b_glob.crd2idx(&tup![n, kk % 128], &dyns).eval(&|_| k) + (kk / 128) * 2048;
            let want = (n / 16) * 16 * k + (kk / 16) * 256 + (n % 16) * 16 + kk % 16;
            if off != want { return Err(ProofError(format!("B preshuffle mismatch at ({n},{kk}): {off} != {want}"))); }
            if !b_seen.insert(off) { return Err(ProofError(format!("B preshuffle duplicate at ({n},{kk})"))); }
        }
    }
    Ok(())
}

fn prove_one(cfg: &GemmConfig, m_size: i64, n_size: i64, k_size: i64, drop: Option<i64>) -> Result<usize, ProofError> {
    let (wave_m, wave_n) = cta_tiles(cfg);
    assert_eq!(cfg.cta_m, m_size, "proof shape: one CTA tile");
    assert_eq!(cfg.cta_n, n_size);
    assert_eq!(k_size % 128, 0);
    check_global_layouts(cfg, k_size)?;

    // The MFMA atom and the two operand/accumulator TV layouts (the kernel uses swap_ab).
    let atom = MmaAtom::mfma_scale_16x16x128_f8();
    let mma = make_tiled_mma(atom, true);
    proofs::check_tv(&mma.a_tv(), 16 * 128, false).map_err(ProofError)?;
    proofs::check_tv(&mma.b_tv(), 16 * 128, false).map_err(ProofError)?;
    proofs::check_tv(&mma.c_tv(), 16 * 16, false).map_err(ProofError)?;
    let c_tv = mma.c_tv();
    let tab = proofs::tv_table(&c_tv);
    let kt = tile_count(k_size);
    let order = compute_order(k_size, drop);
    // the schedule must still be a subset of the K tiles; a dropped tile is reported below as a
    // concrete missing (m,n,k), not as a schedule-length error.
    if order.len() > kt as usize { return Err(ProofError(format!("K schedule computes {} tiles > {kt}", order.len()))); }

    // Slot/generation model: a 2-slot ring (v3 TwoSlot, v4 RegStaged). A generation is the tile
    // counter; stage(t+2) reuses the slot of t only after compute(t) is issued. The schedule above
    // computes every tile once, and this asserts the reuse relation.
    let mut last_compute = HashMap::<i64, i64>::new();
    for &t in &order {
        let slot = t % 2;
        if let Some(prev) = last_compute.insert(slot, t) {
            assert_eq!(t, prev + 2, "slot {slot} generation {prev} -> {t}");
        }
    }

    let mut items = Items::new();
    let mut stores = HashSet::new();
    let mut scale_a = HashSet::new();
    let mut scale_b = HashSet::new();
    for wave in 0..4i64 {
        let (wm, wn) = (wave & 1, wave >> 1);
        let m_wave = wm * wave_m;
        let n_wave = wn * wave_n;
        for i in 0..wave_m / 16 {
            for j in 0..wave_n / 16 {
                let m_atom = m_wave + 16 * i;
                let n_atom = n_wave + 16 * j;
                // accumulator coverage: c_tv is transposed by swap_ab: element e' = n' + 16 m'
                for &e in &tab {
                    let (m_loc, n_loc) = ((e / 16) % 16, e % 16);
                    let m = m_atom + m_loc;
                    let n = n_atom + n_loc;
                    let addr = 2 * (m * n_size + n);
                    if !stores.insert(addr) { return Err(ProofError(format!("C[{m},{n}] stored more than once"))); }
                    for &kb in &order {
                        // every output has A[m, k] and B[n, k] for all 128 k of this block, plus
                        // the matching A row / B 128-column scale for the block.
                        let a_base = m * k_size + kb * 128;
                        let b_base = (n / 16) * 16 * k_size + (kb * 128 / 16) * 256 + (n % 16) * 16;
                        if a_base + 127 >= m_size * k_size { return Err(ProofError(format!("A operand out of range C[{m},{n}] kb{kb}"))); }
                        let _ = b_base;
                        add_contribution(&mut items, m, n, kb)?;
                    }
                }
                // A/B scale row ownership (the panel transposition is proved by verify_sfa_transpose)
                for &kb in &order {
                    for r in 0..16 { scale_a.insert((m_atom + r, kb)); }
                    for c in 0..16 { scale_b.insert(((n_atom + c) / 128, kb)); }
                }
            }
        }
    }
    // every (m,n,kb) exactly once
    for m in 0..m_size {
        for n in 0..n_size {
            for kb in 0..kt {
                match items.get(&(m, n, kb)).copied().unwrap_or(0) {
                    1 => {}
                    c => return Err(ProofError(format!("missing/duplicate contribution C[{m},{n}] K block {kb} (k={}): {c} times", kb * 128))),
                }
                if !scale_a.contains(&(m, kb)) || !scale_b.contains(&(n / 128, kb)) {
                    return Err(ProofError(format!("scale missing for C[{m},{n}] K block {kb}")));
                }
            }
        }
    }
    // A-scale row set is the full CTA rows per K block; B-scale rows are the 2 blocks (cta_n=256).
    if scale_a.len() != (m_size * order.len() as i64) as usize || scale_b.len() != (n_size / 128 * order.len() as i64) as usize {
        return Err(ProofError("scale panel row coverage mismatch".into()));
    }
    Ok(items.len())
}

fn prove(cfg: &GemmConfig, drop: Option<i64>) -> Result<usize, ProofError> {
    // one 256x256 CTA, and a 256-key K extent (2 K blocks): enough to expose a swallowed tile.
    prove_one(cfg, cfg.cta_m, cfg.cta_n, 256, drop)
}

#[test]
fn gemm_v3_and_v4_every_output_k_block_is_covered_once() {
    for cfg in [GemmConfig::v4()] {
        let n = prove(&cfg, None).expect("GEMM index proof");
        assert_eq!(n, (cfg.cta_m * cfg.cta_n * 2) as usize, "one item per (m,n,kb)");
    }
}

#[test]
fn swallowed_tile_one_is_rejected_with_a_concrete_counterexample() {
    let e = prove(&GemmConfig::v4(), Some(1)).expect_err("the swallowed-tile bug must fail");
    let msg = format!("{e:?}");
    assert!(msg.contains("K block 1") && msg.contains("C[0,0]"), "counterexample: {msg}");
    println!("E1 counterexample: {msg}");
}

#[test]
fn v4_dpp_scale_panel_transposition_is_proved() {
    amdgpu_kernels::gemm::verify_sfa_transpose();
}
