//! The kernel descriptor must declare at least as many SGPRs / VGPRs as the instruction stream uses: the hardware does not
//! guarantee registers above `.amdhsa_next_free_{s,v}gpr`. (A regalloc bug once under-declared the SGPR count of the DMA FMHA
//! kernel; its head-1 blocks silently produced zeros.) Checked on the own backend for every registered kernel.
use amdgpu_compiler::{compile, kernel_by_name, kernel_names};

fn highest_registers(asm: &str) -> (i64, i64) {
    let mut max_s: i64 = -1;
    let mut max_v: i64 = -1;
    for line in asm.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('.') || l.starts_with("//") || l.ends_with(':') {
            continue;
        }
        let b = l.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            let prev_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
            if prev_ok && (c == b's' || c == b'v') {
                let mut j = i + 1;
                let mut hi: Option<i64> = None;
                if j < b.len() && b[j] == b'[' {
                    // v[a:b] / s[a:b]
                    let rest = &l[j + 1..];
                    if let Some(end) = rest.find(']') {
                        let inner = &rest[..end];
                        let mut parts = inner.split(':');
                        if let (Some(a), Some(bb)) = (parts.next(), parts.next()) {
                            if let (Ok(_), Ok(bn)) = (a.trim().parse::<i64>(), bb.trim().parse::<i64>()) {
                                hi = Some(bn);
                            }
                        }
                        j += end + 2;
                    }
                } else {
                    let st = j;
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    let next_ok = j >= b.len() || !(b[j].is_ascii_alphanumeric() || b[j] == b'_');
                    if j > st && next_ok {
                        hi = l[st..j].parse::<i64>().ok();
                    }
                }
                if let Some(h) = hi {
                    if c == b's' {
                        max_s = max_s.max(h);
                    } else {
                        max_v = max_v.max(h);
                    }
                }
                i = j.max(i + 1);
                continue;
            }
            i += 1;
        }
    }
    (max_s, max_v)
}

fn declared(asm: &str, key: &str) -> Option<i64> {
    asm.lines().find_map(|l| {
        let l = l.trim();
        l.strip_prefix(key).and_then(|r| r.trim().parse::<i64>().ok())
    })
}

#[test]
fn declared_register_counts_cover_the_instruction_stream() {
    let mut bad = Vec::new();
    for name in kernel_names() {
        let f = kernel_by_name(name).unwrap();
        let Ok(c) = compile(&f) else { continue };
        let (ms, mv) = highest_registers(&c.asm);
        let ds = declared(&c.asm, ".amdhsa_next_free_sgpr").unwrap_or(0);
        let dv = declared(&c.asm, ".amdhsa_next_free_vgpr").unwrap_or(0);
        if ms + 1 > ds {
            bad.push(format!("{name}: SGPR s{ms} used but .amdhsa_next_free_sgpr = {ds}"));
        }
        if mv + 1 > dv {
            bad.push(format!("{name}: VGPR v{mv} used but .amdhsa_next_free_vgpr = {dv}"));
        }
    }
    assert!(bad.is_empty(), "under-declared registers:\n{}", bad.join("\n"));
}