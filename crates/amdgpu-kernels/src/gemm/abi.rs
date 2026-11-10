//! GEMM kernarg ABI and host contracts (dense, conv, MoE).

use super::*;

/// The 96-byte ABI (opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950), declared in offset order.
pub(crate) fn declare_abi(b: &mut FnBuilder) {
    let ptrs = [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)];
    let ints = [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)];
    let mut all: Vec<(&str, u32, bool)> = ptrs.iter().map(|&(n, o)| (n, o, true)).chain(ints.iter().map(|&(n, o)| (n, o, false))).collect();
    all.sort_by_key(|x| x.1);
    for (name, off, is_ptr) in all {
        if is_ptr {
            b.arg(name, DType::Ptr, off, 8)
        } else {
            b.arg(name, DType::I32, off, 4)
        }
    }
}

/// The host contract of the ABI for a `cta_m x cta_n` CTA (what `gpu-verify` proves the accesses against): M = MT cta_m, N = NT cta_n, K = KT 128
/// (the kernel has no partial tiles), the grid (NT, MT), dense strides (stride_a = stride_b = K, stride_c = N, stride_sfa = M, stride_sfb = K / 128),
/// batch 1, and the tensor extents A M K, B N K (preshuffled, same size), C 2 M N, sa (K / 128) M, sb (N / 128) (K / 128) bytes.
pub(crate) fn declare_contract(b: &mut FnBuilder, cta_m: i64, cta_n: i64, scales: Scales, per_tile: i64, split: bool, moe: Option<MoeStage>, conv: Option<ConvCfg>, batched: bool, per_tensor: bool, a8w4: usize, moe_bias: bool) {
    if let Some(cc) = conv {
        return declare_contract_conv(b, cta_m, cta_n, cc);
    }
    if let Some(st) = moe {
        return declare_contract_moe(b, cta_m, cta_n, scales, per_tile, st, moe_bias, a8w4);
    }
    let (mt, nt) = (Ext::sym("MT"), Ext::sym("NT"));
    // int32 offsets: M K, N K and 2 M N stay below 2^31 (16384 x 65536 tiles of bytes at most)
    b.sym_range("MT", 1, (1 << 14) / cta_m);
    b.sym_range("NT", 1, (1 << 14) / cta_n);
    // K <= 128 PANEL_KB: the scale panel holds every K group of the CTA (the vendor sp64 limit); split-K: KT = S KS
    let kt = if split {
        b.sym_range("S", 1, 16);
        b.sym_range("KS", 1, PANEL_KB);
        Ext::sym("S") * Ext::sym("KS")
    } else {
        b.sym_range("KT", 1, PANEL_KB);
        Ext::sym("KT")
    };
    let m = mt.clone() * Ext::c(cta_m);
    let n = nt.clone() * Ext::c(cta_n);
    // batched: BT batches, grid z = BT, every tensor BT times its per-batch extent (the per-batch base checks are ptr_mad's)
    let bt = if batched {
        b.sym_range("BT", 1, 4096);
        b.grid_is(2, Ext::sym("BT"));
        Ext::sym("BT")
    } else {
        Ext::c(1)
    };
    // K in elements (kernarg) and in bytes (extents): KT tiles of 128 bytes
    let k = kt.clone() * Ext::c(128);
    let k_el = kt.clone() * Ext::c(per_tile);
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, if split { Ext::sym("S") } else if batched { Ext::sym("BT") } else { Ext::c(1) });
    if split {
        b.ka_is(88, Ext::sym("KS"));
        b.ka_is(92, m.clone());
        b.grid_is(2, Ext::sym("S"));
    }
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n.clone());
    b.ka_is(80, m.clone());
    b.ka_is(84, kt.clone());
    b.grid_is(0, nt.clone());
    b.grid_is(1, mt.clone());
    // (mixed precision: A has a_images 128-byte images per 128-byte B tile)
    b.arg_tensor("ptr_a", bt.clone() * m.clone() * k.clone() * Ext::c(a8w4 as i64));
    b.arg_tensor("ptr_b", bt.clone() * n.clone() * k.clone());
    if split {
        b.arg_tensor("ptr_c", bt.clone() * Ext::c(4) * Ext::sym("S") * m.clone() * n.clone());
    } else {
        b.arg_tensor("ptr_c", bt.clone() * Ext::c(2) * m.clone() * n.clone());
    }
    match scales {
        Scales::Mx128 => {
            b.arg_tensor("ptr_sfa", bt.clone() * kt.clone() * m);
            b.arg_tensor("ptr_sfb", bt.clone() * nt * Ext::c(cta_n / 128) * kt);
        }
        Scales::RowCol if per_tensor => {
            b.arg_tensor("ptr_sfa", bt.clone() * Ext::c(4));
            b.arg_tensor("ptr_sfb", bt.clone() * Ext::c(4));
        }
        Scales::RowCol => {
            b.arg_tensor("ptr_sfa", bt.clone() * Ext::c(4) * m);
            b.arg_tensor("ptr_sfb", bt.clone() * Ext::c(4) * n);
        }
        Scales::None => {}
        Scales::Mx32 => {
            b.arg_tensor("ptr_sfa", bt.clone() * m * kt.clone() * Ext::c(8));
            b.arg_tensor("ptr_sfb", bt.clone() * n * kt * Ext::c(8));
        }
        Scales::Block128 => {
            b.arg_tensor("ptr_sfa", bt.clone() * m * kt.clone() * Ext::c(4));
            b.arg_tensor("ptr_sfb", bt.clone() * nt * Ext::c(cta_n / 128) * kt * Ext::c(4));
        }
    }
}

/// The implicit-GEMM conv contract (bf16): M = NB OD OH OW rows (the last M tile partial), N = NT cta_n output channels, C = 64 CB input
/// channels, K = C T R S; tensors: input NB DI H W C, weights N K, output M N (bf16).
pub(crate) fn declare_contract_conv(b: &mut FnBuilder, cta_m: i64, cta_n: i64, cc: ConvCfg) {
    for (n, o) in [("NB", 96u32), ("DI", 100), ("H", 104), ("W", 108), ("C", 112), ("OD", 116), ("OH", 120), ("OW", 124)] {
        b.arg(n, DType::I32, o, 4);
    }
    for o in [96u32, 100, 104, 108, 116, 120, 124] {
        b.ka_range(o, 1, 4096);
    }
    b.sym_range("CB", 1, 64);
    b.sym_range("NT", 1, (1 << 14) / cta_n);
    let cb = Ext::sym("CB");
    b.ka_is(112, cb.clone() * Ext::c(64));
    let trs = (cc.t * cc.r * cc.s) as i64;
    let m = Ext::ka(96) * Ext::ka(116) * Ext::ka(120) * Ext::ka(124);
    let n = Ext::sym("NT") * Ext::c(cta_n);
    let kt = cb.clone() * Ext::c(trs);
    let k_el = kt.clone() * Ext::c(64);
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, Ext::c(1));
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n.clone());
    b.ka_is(80, m.clone());
    b.ka_is(84, kt.clone());
    b.grid_is(0, Ext::sym("NT"));
    b.grid_is(1, m.clone().ceil_div(cta_m));
    b.arg_tensor("ptr_a", Ext::ka(96) * Ext::ka(100) * Ext::ka(104) * Ext::ka(108) * cb * Ext::c(128));
    b.arg_tensor("ptr_b", n.clone() * kt * Ext::c(128));
    b.arg_tensor("ptr_c", m * n * Ext::c(2));
}

/// The MoE contract: MT = max M tiles (grid y), NT = output N tiles (grid x), A rows m (tokens or tokens x topk), B rows per expert n, E
/// experts, KT 128-byte K tiles.
pub(crate) fn declare_contract_moe(b: &mut FnBuilder, cta_m: i64, cta_n: i64, scales: Scales, per_tile: i64, st: MoeStage, moe_bias: bool, a_images: usize) {
    if moe_bias {
        b.arg("ptr_bias", DType::Ptr, 144, 8);
    }
    for (n, o) in [("ptr_sorted_ids", 96u32), ("ptr_sorted_expert_ids", 104), ("ptr_num_valid_ids", 112), ("ptr_sorted_weights", 120)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    for (n, o) in [("topk", 128u32), ("E", 132), ("n_out", 136), ("M_tokens", 140)] {
        b.arg(n, DType::I32, o, 4);
    }
    let (mt, nt, kt, ee) = (Ext::sym("MT"), Ext::sym("NT"), Ext::sym("KT"), Ext::sym("E"));
    let nt_c = nt.clone();
    b.sym_range("MT", 1, 4096);
    b.sym_range("NT", 1, 256);
    b.sym_range("KT", 1, PANEL_KB);
    b.sym_range("E", 1, 512);
    b.sym_range("AR", 1, 1 << 20);
    b.ka_range(128, 1, 16);
    b.ka_range(140, 1, 1 << 20);
    let k = kt.clone() * Ext::c(128);
    let k_el = kt.clone() * Ext::c(per_tile);
    // output columns per CTA: stage 1 half the B rows (gate / up interleaved)
    let ncta = if matches!(st, MoeStage::Up { .. }) { cta_n / 2 } else { cta_n };
    let n_out = nt.clone() * Ext::c(ncta);
    let n = if matches!(st, MoeStage::Up { .. }) { n_out.clone() * Ext::c(2) } else { n_out.clone() };
    let m = Ext::sym("AR");
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, Ext::c(1));
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n_out.clone());
    b.ka_is(84, kt.clone());
    b.ka_is(132, ee.clone());
    b.ka_is(136, n_out.clone());
    b.grid_is(0, nt);
    b.grid_is(1, mt.clone());
    // (mixed precision: a_images A images of 128 bytes per 128-byte B tile)
    b.arg_tensor("ptr_a", m.clone() * k.clone() * Ext::c(a_images as i64));
    b.arg_tensor("ptr_b", ee.clone() * n.clone() * k);
    if moe_bias {
        b.arg_tensor("ptr_bias", ee.clone() * n.clone() * Ext::c(4));
    }
    let out_rows = Ext::ka(140) * Ext::ka(128);
    b.arg_tensor("ptr_c", out_rows * n_out * Ext::c(if matches!(st, MoeStage::Up { .. } | MoeStage::DownH) { 2 } else { 4 }));
    b.arg_tensor("ptr_sorted_ids", mt.clone() * Ext::c(4 * cta_m));
    b.arg_tensor("ptr_sorted_weights", mt.clone() * Ext::c(4 * cta_m));
    b.arg_tensor("ptr_sorted_expert_ids", mt * Ext::c(4));
    b.arg_tensor("ptr_num_valid_ids", Ext::c(8));
    if scales == Scales::RowCol {
        b.arg_tensor("ptr_sfa", Ext::c(4) * m);
        b.arg_tensor("ptr_sfb", Ext::c(4) * ee * n);
    } else if scales == Scales::Block128 {
        b.arg_tensor("ptr_sfa", Ext::c(4) * m * kt.clone());
        b.arg_tensor("ptr_sfb", Ext::c(4 * cta_n / 128) * ee * nt_c * kt);
    } else if scales == Scales::Mx32 {
        // MXFP4 MoE: 1 x 32 E8M0 [rows][8 KT] bytes (A rows, every expert's n rows; not shuffled: gathered rows)
        b.arg_tensor("ptr_sfa", Ext::c(8) * m * kt.clone());
        b.arg_tensor("ptr_sfb", Ext::c(8) * ee * n * kt);
    }
}
