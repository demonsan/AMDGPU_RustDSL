//! gfx1250 VGPR MSB windowing (docs/target-gfx1250.md section 3.3.2.3): an instruction encodes VGPR numbers 0..255; register bits [9:8] of
//! each operand slot (dst, src0, src1, src2) come from the MODE MSBs that `s_set_vgpr_msb imm` (imm[1:0] src0, [3:2] src1, [5:4] src2,
//! [7:6] dst) sets. The emitter writes the instruction with the low numbers and puts an `s_set_vgpr_msb` before it when a slot it uses
//! needs other MSBs than the current ones. Convention (as LLVM's): all MSBs are 0 at entry and at every block start; a block whose MSBs
//! are not 0 resets them before its terminator. Slots per encoding (checked against the node disassembler's annotations): VOP*: vdst /
//! src0..2; VGLOBAL: vdst (loads), addr src0, store data src1; VDS: vdst, addr src0, data0 src1, data1 src2; VBUFFER: vdata (loads and
//! stores) dst, vaddr src0. Operands the hardware reads without MSBs (the WMMA scale sources) must be v0..v255.

use amdgpu_target::gfx1250_co::tables::OPS;

/// One VGPR operand: (first register, last register) or None.
fn vgpr_of(tok: &str) -> Option<(u32, u32)> {
    let t = tok.trim().trim_start_matches('-');
    let r = t.strip_prefix('v')?;
    if let Some(inner) = r.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        let (a, b) = inner.split_once(':')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    } else {
        let n: u32 = r.parse().ok()?;
        Some((n, n))
    }
}

fn low(tok: &str, lo: u32, hi: u32) -> String {
    let neg = if tok.trim().starts_with('-') { "-" } else { "" };
    let base = lo % 256;
    if lo == hi {
        format!("{neg}v{base}")
    } else {
        format!("{neg}v[{base}:{}]", base + (hi - lo))
    }
}

/// Slot index (0 dst, 1 src0, 2 src1, 3 src2) of each operand position of an instruction, None = no VGPR slot (SGPR, constant, or an
/// operand the hardware reads without MSBs: Some(4)).
fn slots(mn: &str, n_ops: usize) -> Option<Vec<Option<usize>>> {
    // a DPP form has the slots of its base encoding (the DPP word's source 0 is the src0 slot)
    let mn = mn.strip_suffix("_dpp").unwrap_or(mn);
    let base = mn.strip_suffix("_e32").or_else(|| mn.strip_suffix("_e64")).unwrap_or(mn);
    let fmt = OPS.iter().find(|(m, _, _)| *m == base).map(|x| x.1)?;
    let mut v = vec![None; n_ops];
    match fmt {
        "VOPC" => {
            for (k, s) in v.iter_mut().enumerate().skip(1) {
                *s = Some(k).filter(|k| *k <= 3);
            }
        }
        "VOP1" | "VOP2" | "VOP3" | "VOP3P" | "VOP3PX2" => {
            let carry = matches!(base, "v_add_co_u32" | "v_sub_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32");
            if base != "v_readfirstlane_b32" {
                v[0] = Some(0);
            }
            let mut src = 1;
            for (k, s) in v.iter_mut().enumerate().skip(1) {
                if carry && k == 1 {
                    continue;
                }
                *s = if src <= 3 { Some(src) } else { Some(4) };
                src += 1;
            }
        }
        "VGLOBAL" => {
            if base.starts_with("global_store") {
                v[0] = Some(1);
                if n_ops > 1 {
                    v[1] = Some(2);
                }
            } else if base.starts_with("global_prefetch") {
                v[0] = Some(1);
            } else {
                v[0] = Some(0);
                if n_ops > 1 {
                    v[1] = Some(1);
                }
            }
        }
        "VDS" => {
            if base.starts_with("ds_store") {
                for (k, s) in v.iter_mut().enumerate() {
                    *s = Some(k + 1).filter(|x| *x <= 3);
                }
            } else {
                v[0] = Some(0);
                if n_ops > 1 {
                    v[1] = Some(1);
                }
            }
        }
        "VBUFFER" => {
            v[0] = Some(0);
            if n_ops > 1 {
                v[1] = Some(1);
            }
        }
        _ => {}
    }
    Some(v)
}

/// The window state of the emitter: the MSBs of the four slots (dst, src0, src1, src2).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Msb(pub [u8; 4]);

impl Msb {
    pub fn imm(&self) -> u32 {
        (self.0[1] as u32) | (self.0[2] as u32) << 2 | (self.0[3] as u32) << 4 | (self.0[0] as u32) << 6
    }
}

/// The line rewritten to low VGPR numbers, preceded by an `s_set_vgpr_msb` when its slots need other MSBs than `cur` (updated). `exact`:
/// the author sets the MSBs; a mismatch is an error instead of an insertion.
pub fn window(line: &str, cur: &mut Msb, exact: bool) -> Result<Vec<String>, String> {
    let t = line.trim();
    if let Some(rest) = t.strip_prefix("s_set_vgpr_msb") {
        let imm = rest.trim().trim_start_matches("0x");
        let v = u32::from_str_radix(imm, if rest.trim().starts_with("0x") { 16 } else { 10 }).map_err(|_| format!("`{t}`"))?;
        *cur = Msb([((v >> 6) & 3) as u8, (v & 3) as u8, ((v >> 2) & 3) as u8, ((v >> 4) & 3) as u8]);
        return Ok(vec![line.to_string()]);
    }
    let (mn, rest) = match t.find(' ') {
        Some(i) => (&t[..i], &t[i + 1..]),
        None => return Ok(vec![line.to_string()]),
    };
    if !(mn.starts_with("v_") || mn.starts_with("global_") || mn.starts_with("ds_") || mn.starts_with("buffer_")) {
        return Ok(vec![line.to_string()]);
    }
    // operands are ", "-separated; trailing modifiers follow the last operand after a blank
    let mut ops: Vec<String> = rest.split(", ").map(|s| s.to_string()).collect();
    let mut tail = String::new();
    if let Some(last) = ops.last_mut() {
        if let Some(i) = last.find(' ') {
            tail = last[i..].to_string();
            last.truncate(i);
        }
    }
    if !ops.iter().any(|o| vgpr_of(o).map_or(false, |(_, hi)| hi >= 256)) && cur.0 == [0; 4] {
        return Ok(vec![line.to_string()]);
    }
    let Some(sl) = slots(mn, ops.len()) else { return Err(format!("`{t}`: VGPRs above v255 in an instruction the MSB table does not know")) };
    let mut need: [Option<u8>; 4] = [None; 4];
    for (k, o) in ops.iter_mut().enumerate() {
        let Some((lo, hi)) = vgpr_of(o) else { continue };
        if lo / 256 != hi / 256 {
            return Err(format!("`{t}`: the tuple v[{lo}:{hi}] crosses a 256-register window"));
        }
        match sl.get(k).copied().flatten() {
            Some(s @ 0..=3) => {
                let m = (lo / 256) as u8;
                if need[s].map_or(false, |x| x != m) {
                    return Err(format!("`{t}`: two operands of one slot in different VGPR windows"));
                }
                need[s] = Some(m);
            }
            _ => {
                if lo >= 256 {
                    return Err(format!("`{t}`: operand {k} is read without the VGPR MSBs and must be in v0..v255"));
                }
            }
        }
        *o = low(o, lo, hi);
    }
    let mut out = Vec::new();
    let mut want = *cur;
    for s in 0..4 {
        if let Some(m) = need[s] {
            want.0[s] = m;
        }
    }
    if want != *cur {
        if exact {
            return Err(format!("`{t}`: the VGPR MSBs in effect ({:#x}) do not select its registers (needs {:#x})", cur.imm(), want.imm()));
        }
        out.push(format!("s_set_vgpr_msb {:#x}", want.imm()));
        *cur = want;
    }
    let indent = &line[..line.len() - line.trim_start().len()];
    out.push(format!("{indent}{mn} {}{tail}", ops.join(", ")));
    Ok(out)
}