//! Static VGPR read-bank report for gfx1250 code (the final instruction text). Model (the MI400 register file as the team measured /
//! documented it): the VGPRs live in 8 RAMs round-robin (register % 8; a 256-register window adds a multiple of 8, so the low number the
//! text shows has the same RAM), one read per RAM per cycle; a VALU instruction whose distinct VGPR sources need k reads from the busiest
//! RAM spends k - 1 extra read cycles. Reads served without a RAM access: a source in the wave's 4-entry source cache (the last four
//! distinct VGPRs read; LRU; cleared by an EXEC write or a readlane / writelane) and a source written by one of the previous 4 VALU
//! instructions (result forwarding). Multi-dword sources (64-bit operands, the WMMA fragments) count one read per dword; an 8-aligned
//! 8-register fragment touches every RAM once, so WMMA operand tuples only conflict through their unaligned / single-register parts.
//! VOPD pairs are excluded (their bank rules are checked by GV-1250-VOPD). A heuristic report: it ranks code, it does not time it.

#[derive(Clone, Debug)]
pub struct Entry {
    pub line: usize,
    pub text: String,
    pub extra: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub valu: usize,
    pub conflicted: usize,
    pub extra_cycles: u64,
    pub entries: Vec<Entry>,
}

fn regs(tok: &str) -> Vec<u32> {
    let t = tok.trim().trim_start_matches('-').trim_start_matches('|').trim_end_matches('|');
    if let Some(r) = t.strip_prefix("v[") {
        if let Some((a, b)) = r.trim_end_matches(']').split_once(':') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                return (a..=b).collect();
            }
        }
        return Vec::new();
    }
    match t.strip_prefix('v').and_then(|x| x.parse::<u32>().ok()) {
        Some(r) => vec![r],
        None => Vec::new(),
    }
}

/// The report of one kernel's assembly text.
pub fn report(asm: &str) -> Report {
    let mut rep = Report::default();
    let mut cache: Vec<u32> = Vec::new();
    let mut recent: Vec<Vec<u32>> = Vec::new();
    for (ln, raw) in asm.lines().enumerate() {
        let l = raw.split("//").next().unwrap_or("").trim();
        if l.is_empty() || l.starts_with('.') {
            continue;
        }
        if l.ends_with(':') {
            // a block start: nothing is known about the cache / forwarding paths
            cache.clear();
            recent.clear();
            continue;
        }
        let (mn, ops) = l.split_once(char::is_whitespace).unwrap_or((l, ""));
        if mn.contains("exec") || ops.split(',').next().map_or(false, |d| d.trim().starts_with("exec")) || mn.starts_with("v_readlane") || mn.starts_with("v_writelane") || mn.starts_with("v_cmpx") {
            cache.clear();
        }
        if !mn.starts_with("v_") || ops.contains("::") || mn == "v_nop" {
            continue;
        }
        let o: Vec<&str> = ops.split(',').map(|x| x.trim()).collect();
        if o.is_empty() {
            continue;
        }
        // destination = the first operand (VOPC / cmp writes vcc / an SGPR: no VGPR destination); sources = the rest (an fmac / WMMA C
        // tuple that is also the destination is read too)
        let dst = regs(o[0]);
        let mut srcs: Vec<u32> = Vec::new();
        for s in o.iter().skip(1) {
            // trailing modifiers after a blank (`v4 op_sel:[..]`)
            let s = s.split_whitespace().next().unwrap_or("");
            for r in regs(s) {
                if !srcs.contains(&r) {
                    srcs.push(r);
                }
            }
        }
        if mn.starts_with("v_fmac") || mn.starts_with("v_mac") {
            for r in &dst {
                if !srcs.contains(r) {
                    srcs.push(*r);
                }
            }
        }
        rep.valu += 1;
        // reads that need a RAM access
        let fwd: Vec<u32> = recent.iter().flatten().copied().collect();
        let mut per_ram = [0u32; 8];
        for r in &srcs {
            if cache.contains(r) || fwd.contains(r) {
                continue;
            }
            per_ram[(*r % 8) as usize] += 1;
        }
        let busiest = per_ram.iter().copied().max().unwrap_or(0);
        // a source needing more than one read cycle: with one read per RAM per cycle the reads of an instruction take `busiest` cycles;
        // multi-dword operands are read over several cycles anyway, so only count the excess over an even spread
        let total: u32 = per_ram.iter().sum();
        let even = total.div_ceil(8);
        let extra = busiest.saturating_sub(even.max(1));
        if extra > 0 {
            rep.conflicted += 1;
            rep.extra_cycles += extra as u64;
            rep.entries.push(Entry { line: ln + 1, text: l.to_string(), extra });
        }
        // the source cache (LRU, 4 entries) and the forwarding window (the last 4 VALU results)
        for r in &srcs {
            cache.retain(|x| x != r);
            cache.push(*r);
            if cache.len() > 4 {
                cache.remove(0);
            }
        }
        recent.push(dst);
        if recent.len() > 4 {
            recent.remove(0);
        }
    }
    rep
}

/// A summary line and the worst `top` entries.
pub fn render(name: &str, r: &Report, top: usize) -> String {
    let mut es: Vec<&Entry> = r.entries.iter().collect();
    es.sort_by(|a, b| b.extra.cmp(&a.extra).then(a.line.cmp(&b.line)));
    let mut s = format!("{name}: {} VALU instruction(s), {} with VGPR read-bank conflicts, ~{} extra read cycle(s) (static, per execution of each line)\n", r.valu, r.conflicted, r.extra_cycles);
    for e in es.iter().take(top) {
        s.push_str(&format!("  line {}: +{}: {}\n", e.line, e.extra, e.text));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_ram_sources_cost_extra_cycles() {
        // v0, v8, v16 all in RAM 0: 3 reads from one RAM
        let r = report("v_fma_f32 v11, v0, v8, v16\n");
        assert_eq!(r.extra_cycles, 2, "{r:?}");
        // v0, v9, v18: three RAMs
        assert_eq!(report("v_fma_f32 v11, v0, v9, v18\n").extra_cycles, 0);
        // the source cache: the second instruction reads v0 / v8 from the cache
        let r = report("v_fma_f32 v11, v0, v8, v16\nv_fma_f32 v12, v0, v8, v24\n");
        assert_eq!(r.extra_cycles, 2, "{r:?}");
    }
}
