//! The `_vs128` LDS addressing DERIVED from layouts reproduces the hand-written lane formulas of `fmha_t_build` / `tile::frag` (`kfx`, `trx`) for every
//! lane and issue, with the same number of base registers - and the layout algebra folds into immediates what the hand kernel still adds with VALU.
use amdgpu_layout::proofs::*;
use amdgpu_layout::*;

/// The manual FMHA swizzle s(r) (kernels.rs `swz_s`).
fn swz_s(r: i64) -> i64 {
    let p = r & 1;
    let (u0, u1, u2) = ((r >> 1) & 1, (r >> 2) & 1, (r >> 3) & 1);
    ((u2 ^ p) << 2) + (u0 << 1) + (u1 ^ p)
}

/// A operand of v_mfma_scale_f32_32x32x64_f8f6f4 (bytes; element index m + 32 k): lane l holds row l % 32 and k = 16 (l / 32) + {0..15} and
/// 32 + 16 (l / 32) + {0..15} (the two 16-byte halves of the 32 operand bytes).
fn mfma_a_32x32x64_f8() -> Layout {
    make_layout(tup![tup![32, 2], tup![16, 2]], tup![tup![1, 512], tup![32, 1024]])
}

#[test]
fn k_fragment_reads_reproduce_the_hand_kfx() {
    let mut d = Dyns::new();
    let lane = d.sym("lane", 0, 63);
    let plan = IssuePlan::new(mfma_a_32x32x64_f8(), 16); // ds_read_b128: 16 bytes per issue, 2 issues per fragment
    let mut exprs = Vec::new();
    let mut tags = Vec::new();
    for s in 0..4i64 {
        for c in 0..2i64 {
            // the MFMA operand tile (32 rows x 64 bytes) at rows 32 s, k-group c, inside the [key, d] tile
            let tile = make_layout(tup![32, 64], tup![128, 1]);
            for (i, e) in plan.offsets(&tile, &lane, &Ix::C(4096 * s + 64 * c), &d).into_iter().enumerate() {
                exprs.push(e);
                tags.push((s, c, i as i64));
            }
        }
    }
    let op = layout_to_offsets(&exprs, &Swizzle::fmha128(), ImmRange::DS, &d);
    assert_eq!(op.bases.len(), 4, "four base registers (the hand kfx[0..3]): {:?}", op.bases);
    for (n, &(s, c, i)) in tags.iter().enumerate() {
        let (b, imm) = op.issues[n];
        assert_eq!(imm, 4096 * s, "the row-group offset 4096 s folds into the ds_read immediate (the hand code adds it with v_add)");
        let k = 2 * c + i; // kfx index
        for l in 0..64i64 {
            let (j, h) = (l & 31, l >> 5);
            let kfx = j * 128 + (((h ^ swz_s(j)) ^ (2 * k)) << 4);
            assert_eq!(op.bases[b].eval(&|_| l) + imm, kfx + 4096 * s, "lane {l} s {s} c {c} issue {i}");
        }
    }
}

#[test]
fn hand_k_reads_deliver_the_mfma_a_layout() {
    // inverse direction: un-swizzle the hand addresses and check that lane l / value v holds exactly the element the MFMA A operand expects
    let sw = Swizzle::fmha128();
    for s in 0..4i64 {
        for c in 0..2i64 {
            let mut tab = Vec::new();
            for l in 0..64i64 {
                let (j, h) = (l & 31, l >> 5);
                for v in 0..32i64 {
                    let (i, byte) = (v / 16, v % 16);
                    let k = 2 * c + i;
                    let addr = j * 128 + (((h ^ swz_s(j)) ^ (2 * k)) << 4) + 4096 * s + byte;
                    let lin = sw.apply(addr); // the swizzle is an involution
                    let (row, col) = (lin / 128, lin % 128);
                    let (m, kk) = (row - 32 * s, col - 64 * c);
                    tab.push(m + 32 * kk);
                }
            }
            check_operand_match(&tab, &mfma_a_32x32x64_f8(), 32, None).unwrap();
        }
    }
    // a wrong K order is caught
    let mut bad = Vec::new();
    for l in 0..64i64 {
        for v in 0..32i64 {
            bad.push((l & 31) + 32 * (32 * (l >> 5) + v)); // k contiguous per lane half instead of the two 16-byte halves
        }
    }
    assert!(check_operand_match(&bad, &mfma_a_32x32x64_f8(), 32, None).is_err());
}

/// Source addresses of the transposing V reads (`ds_read_b64_tr_b8`) in tile coordinates (key + 128 dbyte): lane bit 0 -> byte +8, bits 1, 2 -> key
/// +1, +2, bit 3 -> key +8, bit 4 -> byte +16, bit 5 -> key +4; issue modes: read r (key +16 r), half hh (key +64 hh), column group e (byte +32 e).
fn tr_source() -> Layout {
    make_layout(
        tup![tup![2, 2, 2, 2, 2, 2], tup![4, 2, 4]],
        tup![tup![8 * 128, 1, 2, 8, 16 * 128, 4], tup![16, 64, 32 * 128]],
    )
}

#[test]
fn transposed_v_reads_reproduce_the_hand_trx() {
    let mut d = Dyns::new();
    let lane = d.sym("lane", 0, 63);
    let plan = IssuePlan::new(tr_source(), 1); // one 8-byte address per issue
    let tile = make_layout(tup![128, 128], tup![128, 1]);
    let exprs = plan.offsets(&tile, &lane, &Ix::C(0), &d);
    assert_eq!(exprs.len(), 32);
    let op = layout_to_offsets(&exprs, &Swizzle::fmha128(), ImmRange::DS, &d);
    assert_eq!(op.bases.len(), 4, "four base registers (the hand trx[0..3]): {:?}", op.bases);
    for n in 0..32usize {
        let (r, hh, e) = ((n % 4) as i64, ((n / 4) % 2) as i64, (n / 8) as i64);
        let (_, imm) = op.issues[n];
        assert_eq!(imm, 2048 * r + 8192 * hh, "the read / half offsets fold into the immediate");
        let mut addrs = Vec::new();
        for l in 0..64i64 {
            let key_part = 8 * ((l & 15) >> 3) + 4 * (l >> 5) + ((l >> 1) & 3);
            let m_gl = (l >> 4) & 1;
            let z = m_gl ^ swz_s(key_part);
            let trx = key_part * 128 + 8 * (l & 1) + ((z ^ (2 * e)) << 4);
            let got = op.eval(n, &|_| l);
            assert_eq!(got, trx + 8192 * hh + 2048 * r, "lane {l} issue {n}");
            addrs.push(got);
        }
        check_tr_b8_source(&addrs).unwrap();
    }
}

#[test]
fn the_fmha_swizzle_is_bank_conflict_free_where_the_plain_layout_is_not() {
    let model = |sw: &Swizzle| -> (usize, usize) {
        let mut d = Dyns::new();
        let lane = d.sym("lane", 0, 63);
        // K fragment reads: ds_read_b128, 16 lanes per phase
        let plan = IssuePlan::new(mfma_a_32x32x64_f8(), 16);
        let tile = make_layout(tup![32, 64], tup![128, 1]);
        let ks = layout_to_offsets(&plan.offsets(&tile, &lane, &Ix::C(0), &d), sw, ImmRange::DS, &d);
        let mut k_conf = 0;
        for n in 0..ks.issues.len() {
            let a: Vec<i64> = (0..64).map(|l| ks.eval(n, &|_| l)).collect();
            k_conf += bank_conflicts(&a, 16, 64, 16);
        }
        // V transposing reads: 8 bytes, 32 lanes per phase
        let tp = IssuePlan::new(tr_source(), 1);
        let vs = layout_to_offsets(&tp.offsets(&make_layout(tup![128, 128], tup![128, 1]), &lane, &Ix::C(0), &d), sw, ImmRange::DS, &d);
        let mut v_conf = 0;
        for n in 0..vs.issues.len() {
            let a: Vec<i64> = (0..64).map(|l| vs.eval(n, &|_| l)).collect();
            v_conf += bank_conflicts(&a, 8, 64, 32);
        }
        (k_conf, v_conf)
    };
    assert_eq!(model(&Swizzle::fmha128()), (0, 0), "docs/status.md: zero conflicts for the K reads and the transposing V reads");
    let (k, v) = model(&Swizzle::none());
    assert!(k > 0 && v > 0, "the dense 128-byte rows without the swizzle conflict (K {k}, V {v})");
}

#[test]
fn dma_staging_is_contiguous_per_issue_and_covers_the_tile() {
    // one wave stages 32 rows x 128 bytes with 4 x global_load_lds_dwordx4: lane i of DMA q writes LDS bytes 16 i of the q-th KB (linear landing), i.e.
    // row 8 q + (i >> 3), chunk i & 7. TV over the wave's [32 rows x 128 bytes] slab: thread (8 chunks, 8 rows), value (16 bytes, 4 DMAs).
    let tv = make_layout(tup![tup![8, 8], tup![16, 4]], tup![tup![16 * 32, 1], tup![32, 8]]);
    check_tv(&tv, 32 * 128, false).unwrap();
    // LDS side: [row, byte] row-major 128-byte rows; each DMA issue writes 16 consecutive bytes per lane (dwordx4)
    let lds = ComposedLayout::plain(make_layout(tup![32, 128], tup![128, 1]));
    check_contiguous(&tv, &lds, 16, 1, 16).unwrap();
    // with the swizzle on the LDS side the issue stays contiguous (the swizzle only moves whole 16-byte chunks)
    check_contiguous(&tv, &ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 128], tup![128, 1])), 16, 1, 16).unwrap();
    // an 8-byte-chunk swizzle would split a 16-byte vector: rejected
    let bad = ComposedLayout::new(Swizzle::cute(3, 3, 4), make_layout(tup![32, 128], tup![128, 1]));
    assert!(check_contiguous(&tv, &bad, 16, 1, 16).is_err());
}

#[test]
fn mfma_c_layout_is_the_keymask_row_formula() {
    // C of the 32x32 MFMA (element row + 32 col): lane l value r -> row 8 (r / 4) + 4 (l / 32) + r % 4, column l % 32 (tile::reduce::KeyMask)
    let c = make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]]);
    check_tv(&c, 1024, false).unwrap();
    let tab = tv_table(&c);
    for l in 0..64i64 {
        for r in 0..16i64 {
            assert_eq!(tab[(l * 16 + r) as usize], (8 * (r / 4) + 4 * (l / 32) + r % 4) + 32 * (l % 32));
        }
    }
}

#[test]
fn operand_match_with_a_k_permutation() {
    // a reader that swaps the two 16-byte halves of every 32-byte k block: matches the MFMA A layout only under the same K permutation
    // (logical k = k0 + 16 a + 32 b  ->  physical k0 + 32 a + 16 b)
    let perm = make_layout(tup![16, 2, 2], tup![1, 32, 16]);
    let atom = mfma_a_32x32x64_f8();
    let tab: Vec<i64> = tv_table(&atom).into_iter().map(|e| (e % 32) + 32 * perm.at(e / 32)).collect();
    check_operand_match(&tab, &atom, 32, Some(&perm)).unwrap();
    assert!(check_operand_match(&tab, &atom, 32, None).is_err(), "without the permutation the layouts differ");
}
