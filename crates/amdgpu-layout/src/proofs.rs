//! Build-time layout proofs (cute-layout style). They run when a kernel is BUILT (= the GPU program's compile time) and fail the build with the offending
//! layout and coordinate. They are redundant early checks: `gpu_verify` still proves every access on the IR; nothing here is a trust source.

use crate::layout::Layout;
use crate::swizzle::ComposedLayout;

/// The table `tv(t, v)` for all threads and values (static TV layout), indexed `[t * V + v]`.
pub fn tv_table(tv: &Layout) -> Vec<i64> {
    let (t, v) = (tv.get(0).size().st(), tv.get(1).size().st());
    let (lt, lv) = (tv.get(0), tv.get(1));
    let mut out = Vec::with_capacity((t * v) as usize);
    for ti in 0..t {
        for vi in 0..v {
            out.push(lt.at(ti) + lv.at(vi));
        }
    }
    out
}

/// COVERAGE: every element of the `tile`-element tile is reached by some (thread, value), and nothing outside it is. With `dup = false` additionally
/// INJECTIVE (no element twice); `dup = true` allows a declared broadcast (every element reached the same number of times).
pub fn check_tv(tv: &Layout, tile: i64, dup: bool) -> Result<(), String> {
    let tab = tv_table(tv);
    let mut hits = vec![0u32; tile as usize];
    for (k, &e) in tab.iter().enumerate() {
        if e < 0 || e >= tile {
            let v = tv.get(1).size().st();
            return Err(format!("TV layout {tv}: (thread {}, value {}) maps to element {e}, outside the {tile}-element tile", k as i64 / v, k as i64 % v));
        }
        hits[e as usize] += 1;
    }
    if let Some(e) = hits.iter().position(|&h| h == 0) {
        return Err(format!("TV layout {tv}: element {e} of the {tile}-element tile is not covered"));
    }
    let h0 = hits[0];
    if !dup {
        if let Some(e) = hits.iter().position(|&h| h > 1) {
            return Err(format!("TV layout {tv}: element {e} is reached {} times (not injective)", hits[e]));
        }
    } else if let Some(e) = hits.iter().position(|&h| h != h0) {
        return Err(format!("TV layout {tv}: broadcast is not uniform (element 0 x{h0}, element {e} x{})", hits[e]));
    }
    Ok(())
}

/// CONTIGUITY / VECTOR WIDTH: for every thread and issue, the `vec` values of the issue land on `vec * elem_bytes` consecutive bytes of `dst`, aligned to
/// `align` bytes (e.g. 16 for a b128 access).
pub fn check_contiguous(tv: &Layout, dst: &ComposedLayout, vec: i64, elem_bytes: i64, align: i64) -> Result<(), String> {
    let (t, v) = (tv.get(0).size().st(), tv.get(1).size().st());
    let (lt, lv) = (tv.get(0), tv.get(1));
    for ti in 0..t {
        for i in 0..v / vec {
            let first = dst.at(lt.at(ti) + lv.at(i * vec));
            if first % align != 0 {
                return Err(format!("thread {ti} issue {i}: byte offset {first} is not {align}-byte aligned"));
            }
            for k in 1..vec {
                let o = dst.at(lt.at(ti) + lv.at(i * vec + k));
                if o != first + k * elem_bytes {
                    return Err(format!("thread {ti} issue {i}: value {k} lands at byte {o}, not {} (the issue is not one contiguous vector)", first + k * elem_bytes));
                }
            }
        }
    }
    Ok(())
}

/// `ds_read_b64_tr_b8` SOURCE validity for one issue: every lane's address is 8-byte aligned and the 64 lanes read 64 distinct quad-words. (The register
/// side - which bytes each lane receives - is the atom's documented layout, pinned to the hardware probe `probe_lds_tr_b8` in the atom tables.)
pub fn check_tr_b8_source(addrs: &[i64]) -> Result<(), String> {
    for (l, a) in addrs.iter().enumerate() {
        if a % 8 != 0 {
            return Err(format!("ds_read_b64_tr_b8: lane {l} address {a} is not 8-byte aligned"));
        }
    }
    let mut s = addrs.to_vec();
    s.sort_unstable();
    if let Some(w) = s.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("ds_read_b64_tr_b8: two lanes read the same quad-word at {}", w[0]));
    }
    Ok(())
}

/// MFMA OPERAND MATCH: a register TV table produced by a copy (thread -> its values' elements of the operand tile `m + M k`) equals the MMA atom's operand TV
/// layout, after the K permutation `kperm` (logical k -> physical k; `None` = identity) is applied to the atom's elements.
pub fn check_operand_match(copy_tab: &[i64], mma_tv: &Layout, m_rows: i64, kperm: Option<&Layout>) -> Result<(), String> {
    let want = tv_table(mma_tv);
    if want.len() != copy_tab.len() {
        return Err(format!("operand match: the copy produces {} (thread, value) pairs, the MMA operand has {}", copy_tab.len(), want.len()));
    }
    let v = mma_tv.get(1).size().st();
    for (k, (&got, &w)) in copy_tab.iter().zip(want.iter()).enumerate() {
        let w = match kperm {
            None => w,
            Some(p) => (w % m_rows) + m_rows * p.at(w / m_rows),
        };
        if got != w {
            return Err(format!(
                "operand match: thread {} value {} holds element (m {}, k {}) but the MMA expects (m {}, k {})",
                k as i64 / v,
                k as i64 % v,
                got % m_rows,
                got / m_rows,
                w % m_rows,
                w / m_rows
            ));
        }
    }
    Ok(())
}

/// LDS bank conflicts of one access: `addrs[lane]` byte addresses of `width` bytes each; the hardware serves `lanes_per_phase` lanes per cycle over `banks`
/// 4-byte banks (gfx950 model: 64 banks; b128 = 16 lanes per phase, b64 = 32, b32 = 64). Returns the extra cycles (0 = conflict free). Lanes reading the
/// same dword are a broadcast, not a conflict.
pub fn bank_conflicts(addrs: &[i64], width: i64, banks: i64, lanes_per_phase: usize) -> usize {
    let mut extra = 0usize;
    for phase in addrs.chunks(lanes_per_phase) {
        let mut per_bank: Vec<Vec<i64>> = vec![Vec::new(); banks as usize];
        for a in phase {
            for w in 0..(width / 4) {
                let dw = a / 4 + w;
                let b = dw.rem_euclid(banks) as usize;
                if !per_bank[b].contains(&dw) {
                    per_bank[b].push(dw);
                }
            }
        }
        let worst = per_bank.iter().map(|v| v.len()).max().unwrap_or(0);
        extra += worst.saturating_sub(1);
    }
    extra
}

/// SWIZZLE: bijectivity on a block of `bytes` (exhaustive).
pub fn check_swizzle_bijective(s: &crate::swizzle::Swizzle, bytes: i64) -> Result<(), String> {
    let mut seen = vec![false; bytes as usize];
    for x in 0..bytes {
        let y = s.apply(x);
        if y < 0 || y >= bytes {
            return Err(format!("swizzle maps {x} to {y}, outside the {bytes}-byte block"));
        }
        if seen[y as usize] {
            return Err(format!("swizzle is not injective: {y} reached twice"));
        }
        seen[y as usize] = true;
    }
    Ok(())
}
