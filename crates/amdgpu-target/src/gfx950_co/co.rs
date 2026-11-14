//! Code-object writer: the compiler's assembly text -> the ELF code object `clang -x assembler ... -mcode-object-version=6` + `ld.lld` would produce, byte for byte (the layout is lld's for one
//! kernel: headers, `.note` (AMDGPU metadata, msgpack), `.dynsym`, `.gnu.hash`, `.hash`, `.dynstr`, `.text` (code + kernel descriptor), `.dynamic`, `.relro_padding`, `.comment`, `.symtab`, ...).

use super::enc::encode_line;
use std::collections::HashMap;

/// The linker identification string of the toolchain this layout was derived from (`.comment`).
pub const LINKER_COMMENT: &str = "Linker: AMD LLD 24.0.0 (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)";

// ------------------------------------------------------------------------------------------------------------------------------ the assembly text

struct Parsed {
    name: String,
    /// (is_label, text)
    items: Vec<(bool, String)>,
    amdhsa: HashMap<String, u64>,
    meta: Vec<String>,
}

fn parse_asm(asm: &str) -> Result<Parsed, String> {
    let mut name = String::new();
    let mut items = Vec::new();
    let mut amdhsa = HashMap::new();
    let mut meta = Vec::new();
    let (mut in_code, mut in_kd, mut in_meta) = (false, false, false);
    for raw in asm.lines() {
        let l = raw.split("//").next().unwrap_or("").trim();
        if in_meta {
            if l == ".end_amdgpu_metadata" {
                in_meta = false;
            } else if raw.trim() != "---" && raw.trim() != "..." {
                meta.push(raw.trim_end().to_string());
            }
            continue;
        }
        if l.is_empty() {
            continue;
        }
        if l == ".amdgpu_metadata" {
            in_meta = true;
            continue;
        }
        if let Some(n) = l.strip_prefix(".globl ") {
            name = n.trim().to_string();
            continue;
        }
        if l.starts_with(".amdhsa_kernel") {
            in_kd = true;
            in_code = false;
            continue;
        }
        if l == ".end_amdhsa_kernel" {
            in_kd = false;
            continue;
        }
        if in_kd {
            if let Some(rest) = l.strip_prefix(".amdhsa_") {
                let mut it = rest.split_whitespace();
                let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or("0"));
                amdhsa.insert(k.to_string(), v.parse::<u64>().map_err(|e| format!("{l}: {e}"))?);
            }
            continue;
        }
        if l.starts_with(".size") {
            in_code = false;
            continue;
        }
        if l.starts_with('.') {
            continue;
        }
        if let Some(lab) = l.strip_suffix(':') {
            if !in_code && lab == name {
                in_code = true;
            } else if in_code {
                items.push((true, lab.to_string()));
            }
            continue;
        }
        if in_code {
            items.push((false, l.to_string()));
        }
    }
    if name.is_empty() {
        return Err("no .globl kernel symbol".into());
    }
    Ok(Parsed { name, items, amdhsa, meta })
}

// ------------------------------------------------------------------------------------------------------------------------------ msgpack metadata

#[derive(Debug)]
enum Val {
    Map(Vec<(String, Val)>),
    List(Vec<Val>),
    Int(u64),
    Str(String),
}

fn indent_of(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

fn scalar(s: &str) -> Val {
    let t = s.trim();
    if let Some(q) = t.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        return Val::Str(q.to_string());
    }
    match t.parse::<u64>() {
        Ok(n) => Val::Int(n),
        Err(_) => Val::Str(t.to_string()),
    }
}

/// Parses the YAML subset the compiler prints (block maps and block sequences of scalars / maps) starting at line `*i`, whose indentation is `ind`.
fn parse_block(lines: &[String], i: &mut usize, ind: usize) -> Val {
    let first = lines[*i].trim_start();
    if first.starts_with("- ") || first == "-" {
        let mut v = Vec::new();
        while *i < lines.len() && indent_of(&lines[*i]) == ind && lines[*i].trim_start().starts_with('-') {
            let rest = lines[*i].trim_start()[1..].trim_start().to_string();
            let item_ind = ind + 2;
            if rest.is_empty() {
                *i += 1;
                v.push(parse_block(lines, i, item_ind));
            } else if let Some((k, val)) = split_kv(&rest) {
                // a map whose first entry sits on the dash line
                let mut entries = Vec::new();
                *i += 1;
                entries.push((k.clone(), if val.is_empty() { parse_nested(lines, i, item_ind) } else { scalar(&val) }));
                while *i < lines.len() && indent_of(&lines[*i]) == item_ind && !lines[*i].trim_start().starts_with('-') {
                    let (k, val) = split_kv(lines[*i].trim()).unwrap_or_default();
                    *i += 1;
                    entries.push((k, if val.is_empty() { parse_nested(lines, i, item_ind) } else { scalar(&val) }));
                }
                v.push(Val::Map(entries));
            } else {
                *i += 1;
                v.push(scalar(&rest));
            }
        }
        Val::List(v)
    } else {
        let mut entries = Vec::new();
        while *i < lines.len() && indent_of(&lines[*i]) == ind && !lines[*i].trim_start().starts_with('-') {
            let (k, val) = split_kv(lines[*i].trim()).unwrap_or_default();
            *i += 1;
            entries.push((k, if val.is_empty() { parse_nested(lines, i, ind) } else { scalar(&val) }));
        }
        Val::Map(entries)
    }
}

fn parse_nested(lines: &[String], i: &mut usize, parent: usize) -> Val {
    if *i < lines.len() && (indent_of(&lines[*i]) > parent || (indent_of(&lines[*i]) == parent && lines[*i].trim_start().starts_with('-'))) {
        let ind = indent_of(&lines[*i]);
        parse_block(lines, i, ind)
    } else {
        Val::List(Vec::new())
    }
}

fn split_kv(s: &str) -> Option<(String, String)> {
    let (k, v) = s.split_once(':')?;
    Some((k.trim().to_string(), v.trim().to_string()))
}

fn mp_uint(out: &mut Vec<u8>, n: u64) {
    match n {
        0..=127 => out.push(n as u8),
        128..=255 => out.extend([0xcc, n as u8]),
        256..=65535 => {
            out.push(0xcd);
            out.extend((n as u16).to_be_bytes());
        }
        65536..=0xffff_ffff => {
            out.push(0xce);
            out.extend((n as u32).to_be_bytes());
        }
        _ => {
            out.push(0xcf);
            out.extend(n.to_be_bytes());
        }
    }
}

fn mp_str(out: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    match b.len() {
        0..=31 => out.push(0xa0 | b.len() as u8),
        32..=255 => out.extend([0xd9, b.len() as u8]),
        _ => {
            out.push(0xda);
            out.extend((b.len() as u16).to_be_bytes());
        }
    }
    out.extend(b);
}

fn mp_val(out: &mut Vec<u8>, v: &Val) {
    match v {
        Val::Int(n) => mp_uint(out, *n),
        Val::Str(s) => mp_str(out, s),
        Val::List(l) => {
            if l.len() < 16 {
                out.push(0x90 | l.len() as u8);
            } else {
                out.push(0xdc);
                out.extend((l.len() as u16).to_be_bytes());
            }
            for x in l {
                mp_val(out, x);
            }
        }
        Val::Map(m) => {
            let mut e: Vec<&(String, Val)> = m.iter().collect();
            e.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            if e.len() < 16 {
                out.push(0x80 | e.len() as u8);
            } else {
                out.push(0xde);
                out.extend((e.len() as u16).to_be_bytes());
            }
            for (k, x) in e {
                mp_str(out, k);
                mp_val(out, x);
            }
        }
    }
}

fn metadata_msgpack(meta: &[String]) -> Vec<u8> {
    let lines: Vec<String> = meta.iter().filter(|l| !l.trim().is_empty()).cloned().collect();
    let mut i = 0;
    let v = parse_block(&lines, &mut i, 0);
    let mut out = Vec::new();
    mp_val(&mut out, &v);
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ kernel descriptor

fn kernel_descriptor(h: &HashMap<String, u64>, kd_off: u64) -> [u8; 64] {
    let g = |k: &str, d: u64| h.get(k).copied().unwrap_or(d);
    let vgprs = g("next_free_vgpr", 0);
    let sgprs = g("next_free_sgpr", 0) + 6; // + vcc, flat_scratch, xnack mask
    let vgran = vgprs.div_ceil(8).saturating_sub(1);
    let sgran = sgprs.div_ceil(8).saturating_sub(1);
    let rsrc1 = vgran | sgran << 6 | g("float_denorm_mode_32", 0) << 16 | g("float_denorm_mode_16_64", 0) << 18 | g("dx10_clamp", 1) << 21 | g("ieee_mode", 1) << 23;
    let rsrc2 = g("user_sgpr_count", 0) << 1 | g("system_sgpr_workgroup_id_x", 0) << 7 | g("system_sgpr_workgroup_id_y", 0) << 8 | g("system_sgpr_workgroup_id_z", 0) << 9 | g("system_vgpr_workitem_id", 0) << 11;
    let rsrc3 = g("accum_offset", 4).div_ceil(4).saturating_sub(1);
    let props = g("user_sgpr_private_segment_buffer", 0) | g("user_sgpr_dispatch_ptr", 0) << 1 | g("user_sgpr_queue_ptr", 0) << 2 | g("user_sgpr_kernarg_segment_ptr", 0) << 3 | g("user_sgpr_dispatch_id", 0) << 4 | g("user_sgpr_flat_scratch_init", 0) << 5 | g("user_sgpr_private_segment_size", 0) << 6;
    let mut kd = [0u8; 64];
    kd[0..4].copy_from_slice(&(g("group_segment_fixed_size", 0) as u32).to_le_bytes());
    kd[4..8].copy_from_slice(&(g("private_segment_fixed_size", 0) as u32).to_le_bytes());
    kd[8..12].copy_from_slice(&(g("kernarg_size", 0) as u32).to_le_bytes());
    kd[16..24].copy_from_slice(&(-(kd_off as i64)).to_le_bytes());
    kd[44..48].copy_from_slice(&(rsrc3 as u32).to_le_bytes());
    kd[48..52].copy_from_slice(&(rsrc1 as u32).to_le_bytes());
    kd[52..56].copy_from_slice(&(rsrc2 as u32).to_le_bytes());
    kd[56..58].copy_from_slice(&(props as u16).to_le_bytes());
    kd
}

// ------------------------------------------------------------------------------------------------------------------------------ ELF

fn align(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

fn sysv_hash(name: &str) -> u32 {
    let mut h: u32 = 0;
    for &c in name.as_bytes() {
        h = (h << 4).wrapping_add(c as u32);
        let g = h & 0xf000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

fn gnu_hash(name: &str) -> u32 {
    let mut h: u32 = 5381;
    for &c in name.as_bytes() {
        h = h.wrapping_mul(33).wrapping_add(c as u32);
    }
    h
}

fn put32(v: &mut Vec<u8>, x: u32) {
    v.extend(x.to_le_bytes());
}
fn put64(v: &mut Vec<u8>, x: u64) {
    v.extend(x.to_le_bytes());
}
fn put16(v: &mut Vec<u8>, x: u16) {
    v.extend(x.to_le_bytes());
}

/// Assembles the compiler's assembly text for one kernel into a code object.
pub fn assemble(asm: &str) -> Result<Vec<u8>, String> {
    let p = parse_asm(asm)?;
    // pass 1: instruction sizes and label offsets (a branch is always one dword)
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut pc = 0usize;
    let mut sizes = Vec::new();
    for (is_label, text) in &p.items {
        if *is_label {
            labels.insert(text.clone(), pc);
            sizes.push(0);
        } else {
            let n = encode_line(text, pc, &|_| Some(pc)).map_err(|e| format!("{text}: {e}"))?.len();
            sizes.push(n);
            pc += n;
        }
    }
    // pass 2: final bytes
    let mut code = Vec::with_capacity(pc);
    let mut label_order: Vec<(String, usize)> = Vec::new();
    for (k, (is_label, text)) in p.items.iter().enumerate() {
        if *is_label {
            label_order.push((text.clone(), code.len()));
            continue;
        }
        let b = encode_line(text, code.len(), &|l| labels.get(l).copied()).map_err(|e| format!("{text}: {e}"))?;
        debug_assert_eq!(b.len(), sizes[k]);
        code.extend(b);
    }
    // the assembler creates a symbol at its first mention (a branch operand or the definition): the symbol table follows that order
    {
        let mut first: Vec<String> = Vec::new();
        for (is_label, text) in &p.items {
            let nm = if *is_label { Some(text.as_str()) } else if text.starts_with("s_branch") || text.starts_with("s_cbranch") { text.split_whitespace().nth(1) } else { None };
            if let Some(n) = nm {
                if !first.iter().any(|x| x == n) {
                    first.push(n.to_string());
                }
            }
        }
        let at: HashMap<&str, usize> = first.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
        label_order.sort_by_key(|(n, _)| at.get(n.as_str()).copied().unwrap_or(usize::MAX));
    }
    let code_len = code.len();
    let kd_off = align(code_len, 64);
    let mut text = code;
    // the alignment fill of a text section is `s_nop 0` (0xbf800000)
    while text.len() < kd_off {
        text.extend(0xbf80_0000u32.to_le_bytes());
    }
    text.extend(kernel_descriptor(&p.amdhsa, kd_off as u64));
    let kd_name = format!("{}.kd", p.name);

    // .note
    let md = metadata_msgpack(&p.meta);
    let mut note = Vec::new();
    put32(&mut note, 7);
    put32(&mut note, md.len() as u32);
    put32(&mut note, 32);
    note.extend(b"AMDGPU\0\0");
    note.extend(&md);
    note.resize(align(note.len(), 4), 0);

    // .dynstr / .dynsym / hashes
    let mut dynstr = vec![0u8];
    let n1 = dynstr.len();
    dynstr.extend(p.name.as_bytes());
    dynstr.push(0);
    let n2 = dynstr.len();
    dynstr.extend(kd_name.as_bytes());
    dynstr.push(0);

    // layout (file offsets; the RX / RW segments sit on their own pages)
    let off_note = 0x200usize;
    let off_dynsym = align(off_note + note.len(), 8);
    let off_gnuhash = align(off_dynsym + 3 * 24, 8);
    let gnuhash_len = 16 + 8 + 4 + 8;
    let off_hash = align(off_gnuhash + gnuhash_len, 4);
    let hash_len = 8 + 3 * 4 + 3 * 4;
    let off_dynstr = align(off_hash + hash_len, 1);
    let off_text = align(off_dynstr + dynstr.len(), 256);
    let seg1_end = off_dynstr + dynstr.len();
    let va_text = align(seg1_end, 0x1000) + off_text % 0x1000;
    let off_dynamic = off_text + text.len();
    let va_dynamic = align(va_text + text.len(), 0x1000) + off_dynamic % 0x1000;
    let dynamic_len = 7 * 16;
    let relro_len = align(va_dynamic + dynamic_len, 0x1000) - (va_dynamic + dynamic_len);
    let off_comment = off_dynamic + dynamic_len;
    let comment: Vec<u8> = {
        let mut c = LINKER_COMMENT.as_bytes().to_vec();
        c.push(0);
        c
    };

    // symbols
    // .strtab: the labels, the two global names, then the linker's `_DYNAMIC`
    let mut strtab = vec![0u8];
    let mut name_at: HashMap<String, u32> = HashMap::new();
    for nm in label_order.iter().map(|l| l.0.as_str()).chain([p.name.as_str(), kd_name.as_str(), "_DYNAMIC"]) {
        if !name_at.contains_key(nm) {
            name_at.insert(nm.to_string(), strtab.len() as u32);
            strtab.extend(nm.as_bytes());
            strtab.push(0);
        }
    }
    let mut symtab = vec![0u8; 24];
    let add_sym = |name: &str, info: u8, other: u8, shndx: u16, value: u64, size: u64, symtab: &mut Vec<u8>| {
        let no = name_at[name];
        put32(symtab, no);
        symtab.push(info);
        symtab.push(other);
        put16(symtab, shndx);
        put64(symtab, value);
        put64(symtab, size);
    };
    for (l, o) in &label_order {
        add_sym(l, 0, 0, 6, va_text as u64 + *o as u64, 0, &mut symtab);
    }
    add_sym("_DYNAMIC", 0, 2, 7, va_dynamic as u64, 0, &mut symtab);
    let nlocal = 1 + label_order.len() + 1;
    add_sym(&p.name, 0x12, 3, 6, va_text as u64, code_len as u64, &mut symtab);
    add_sym(&kd_name, 0x11, 0, 6, (va_text + kd_off) as u64, 64, &mut symtab);

    let mut dynsym = vec![0u8; 24];
    for (nm, info, other, val, size) in [(n1 as u32, 0x12u8, 3u8, va_text as u64, code_len as u64), (n2 as u32, 0x11, 0, (va_text + kd_off) as u64, 64)] {
        put32(&mut dynsym, nm);
        dynsym.push(info);
        dynsym.push(other);
        put16(&mut dynsym, 6);
        put64(&mut dynsym, val);
        put64(&mut dynsym, size);
    }

    // .gnu.hash: one bucket, the two symbols in order
    let hs = [gnu_hash(&p.name), gnu_hash(&kd_name)];
    let mut gh = Vec::new();
    put32(&mut gh, 1); // nbuckets
    put32(&mut gh, 1); // symoffset
    put32(&mut gh, 1); // bloom size
    put32(&mut gh, 26); // bloom shift
    let mut bloom: u64 = 0;
    for h in hs {
        bloom |= 1u64 << (h % 64);
        bloom |= 1u64 << ((h >> 26) % 64);
    }
    put64(&mut gh, bloom);
    put32(&mut gh, 1); // bucket 0 -> the first hashed symbol
    put32(&mut gh, hs[0] & !1);
    put32(&mut gh, hs[1] | 1);

    // .hash (SysV): nbucket = nchain = 3
    let mut sh = Vec::new();
    put32(&mut sh, 3);
    put32(&mut sh, 3);
    let mut buckets = [0u32; 3];
    let mut chains = [0u32; 3];
    for (idx, nm) in [(1u32, &p.name), (2u32, &kd_name)] {
        let b = (sysv_hash(nm) % 3) as usize;
        chains[idx as usize] = buckets[b];
        buckets[b] = idx;
    }
    for b in buckets {
        put32(&mut sh, b);
    }
    for c in chains {
        put32(&mut sh, c);
    }

    // .dynamic
    let mut dynamic = Vec::new();
    for (t, v) in [(6u64, off_dynsym as u64), (11, 24), (5, off_dynstr as u64), (10, dynstr.len() as u64), (0x6fff_fef5, off_gnuhash as u64), (4, off_hash as u64), (0, 0)] {
        put64(&mut dynamic, t);
        put64(&mut dynamic, v);
    }

    // section names
    let names = [".note", ".dynsym", ".gnu.hash", ".hash", ".dynstr", ".text", ".dynamic", ".relro_padding", ".comment", ".symtab", ".shstrtab", ".strtab"];
    let mut shstr = vec![0u8];
    let mut name_off = Vec::new();
    for n in names {
        name_off.push(shstr.len() as u32);
        shstr.extend(n.as_bytes());
        shstr.push(0);
    }
    let off_symtab = align(off_comment + comment.len(), 8);
    let off_shstr = off_symtab + symtab.len();
    let off_strtab = off_shstr + shstr.len();
    let off_sh = align(off_strtab + strtab.len(), 8);

    // assemble the file
    let mut f = vec![0u8; off_sh + 13 * 64];
    let mut w = |off: usize, b: &[u8], f: &mut Vec<u8>| f[off..off + b.len()].copy_from_slice(b);
    // ELF header
    let mut h = Vec::new();
    h.extend([0x7f, b'E', b'L', b'F', 2, 1, 1, 0x40, 4, 0, 0, 0, 0, 0, 0, 0]);
    put16(&mut h, 3);
    put16(&mut h, 0xe0);
    put32(&mut h, 1);
    put64(&mut h, 0);
    put64(&mut h, 64);
    put64(&mut h, off_sh as u64);
    put32(&mut h, 0x54f);
    put16(&mut h, 64);
    put16(&mut h, 56);
    put16(&mut h, 8);
    put16(&mut h, 64);
    put16(&mut h, 13);
    put16(&mut h, 11);
    w(0, &h, &mut f);
    // program headers
    let mut ph = Vec::new();
    let mut seg = |t: u32, fl: u32, off: u64, va: u64, fsz: u64, msz: u64, al: u64, ph: &mut Vec<u8>| {
        put32(ph, t);
        put32(ph, fl);
        put64(ph, off);
        put64(ph, va);
        put64(ph, va);
        put64(ph, fsz);
        put64(ph, msz);
        put64(ph, al);
    };
    let rw_msz = (dynamic_len + relro_len) as u64;
    seg(6, 4, 0x40, 0x40, 0x1c0, 0x1c0, 8, &mut ph);
    let seg1 = (off_dynstr + dynstr.len()) as u64;
    seg(1, 4, 0, 0, seg1, seg1, 0x1000, &mut ph);
    seg(1, 5, off_text as u64, va_text as u64, text.len() as u64, text.len() as u64, 0x1000, &mut ph);
    seg(1, 6, off_dynamic as u64, va_dynamic as u64, dynamic_len as u64, rw_msz, 0x1000, &mut ph);
    seg(2, 6, off_dynamic as u64, va_dynamic as u64, dynamic_len as u64, dynamic_len as u64, 8, &mut ph);
    seg(0x6474e552, 4, off_dynamic as u64, va_dynamic as u64, dynamic_len as u64, rw_msz, 1, &mut ph);
    seg(0x6474e551, 6, 0, 0, 0, 0, 0, &mut ph);
    seg(4, 4, off_note as u64, off_note as u64, note.len() as u64, note.len() as u64, 4, &mut ph);
    w(64, &ph, &mut f);
    w(off_note, &note, &mut f);
    w(off_dynsym, &dynsym, &mut f);
    w(off_gnuhash, &gh, &mut f);
    w(off_hash, &sh, &mut f);
    w(off_dynstr, &dynstr, &mut f);
    w(off_text, &text, &mut f);
    w(off_dynamic, &dynamic, &mut f);
    w(off_comment, &comment, &mut f);
    w(off_symtab, &symtab, &mut f);
    w(off_shstr, &shstr, &mut f);
    w(off_strtab, &strtab, &mut f);
    // section headers
    let mut sec = vec![0u8; 64];
    let mut sh_entry = |name: u32, ty: u32, flags: u64, addr: u64, off: u64, size: u64, link: u32, info: u32, al: u64, es: u64, sec: &mut Vec<u8>| {
        put32(sec, name);
        put32(sec, ty);
        put64(sec, flags);
        put64(sec, addr);
        put64(sec, off);
        put64(sec, size);
        put32(sec, link);
        put32(sec, info);
        put64(sec, al);
        put64(sec, es);
    };
    sh_entry(name_off[0], 7, 2, off_note as u64, off_note as u64, note.len() as u64, 0, 0, 4, 0, &mut sec);
    sh_entry(name_off[1], 11, 2, off_dynsym as u64, off_dynsym as u64, dynsym.len() as u64, 5, 1, 8, 24, &mut sec);
    sh_entry(name_off[2], 0x6fff_fff6, 2, off_gnuhash as u64, off_gnuhash as u64, gh.len() as u64, 2, 0, 8, 0, &mut sec);
    sh_entry(name_off[3], 5, 2, off_hash as u64, off_hash as u64, sh.len() as u64, 2, 0, 4, 4, &mut sec);
    sh_entry(name_off[4], 3, 2, off_dynstr as u64, off_dynstr as u64, dynstr.len() as u64, 0, 0, 1, 0, &mut sec);
    sh_entry(name_off[5], 1, 6, va_text as u64, off_text as u64, text.len() as u64, 0, 0, 256, 0, &mut sec);
    sh_entry(name_off[6], 6, 3, va_dynamic as u64, off_dynamic as u64, dynamic_len as u64, 5, 0, 8, 16, &mut sec);
    sh_entry(name_off[7], 8, 3, (va_dynamic + dynamic_len) as u64, (off_dynamic + dynamic_len) as u64, relro_len as u64, 0, 0, 1, 0, &mut sec);
    sh_entry(name_off[8], 1, 0x30, 0, off_comment as u64, comment.len() as u64, 0, 0, 1, 1, &mut sec);
    sh_entry(name_off[9], 2, 0, 0, off_symtab as u64, symtab.len() as u64, 12, nlocal as u32, 8, 24, &mut sec);
    sh_entry(name_off[10], 3, 0, 0, off_shstr as u64, shstr.len() as u64, 0, 0, 1, 0, &mut sec);
    sh_entry(name_off[11], 3, 0, 0, off_strtab as u64, strtab.len() as u64, 0, 0, 1, 0, &mut sec);
    w(off_sh, &sec, &mut f);
    Ok(f)
}
