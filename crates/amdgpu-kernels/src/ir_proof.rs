//! E3: symbolic execution of the BUILT DSL `Function` for the MXFP8 GEMM.
//!
//! This is not a re-implementation of the layout composition: it walks the actual `Function.stmts`
//! (the same IR `amdc print-ir --stage dsl` prints), unrolls the pipeline loop for a small K
//! (3 K tiles), enumerates the 4 waves / 64 lanes, and abstracts memory as element tags:
//!   * global loads/DMA produce `A[m,k]`, `B[k,n]`, `Sa[m,kb]`, `Sb[block,kb]` tags from the
//!     address expressions the kernel emitted;
//!   * LDS writes/reads move those tags by byte address, scoped by the tile-protocol marker
//!     (tile name, version = generation);
//!   * MFMA statements consume tags from their operand fragments and record a contribution per
//!     accumulator dword / K generation;
//!   * epilogue stores map accumulator dwords back to C addresses and emit the contribution.
//! The final checks are the same theorem as E1, but over the actual kernel IR.
//!
//! The proof-only `GEMM_PROOF_DROP_KB` hook in the kernel adds a real `if` around the MMAs of one
//! tile generation; setting it to 1 reproduces the "tile 1 swallowed" bug in the KERNEL CODE.

use amdgpu_dsl::tile::ev;
use amdgpu_dsl::{Function, Node, Op, Opnd, RegionKind, Stmt, VarId, ValueId};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::gemm_mxfp8::GemmConfig;

#[derive(Debug)]
pub struct ProofError(pub String);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Tag {
    A(i64, i64),
    B(i64, i64),
    Sa(i64, i64),
    Sb(i64, i64),
    /// Accumulator provenance: (VarId, dword index). Carried through the LDS epilogue so a stored
    /// C value can be traced to the exact accumulator dword even across lane transposes.
    Acc(u32, u8),
    U,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PtrRole { A, B, C, Sfa, Sfb }

#[derive(Clone, Debug)]
enum V {
    /// Concrete integer / pointer-offset value.
    Int(i64),
    /// Absolute global pointer: role + byte offset from the tensor start.
    Ptr(PtrRole, i64),
    /// MUBUF descriptor over a tensor.
    Rsrc(PtrRole),
    /// Four bytes of provenance (one dword), selected from a fragment.
    Word { tags: Vec<Tag>, origin: Option<(u32, u8)> },
    /// A fragment of bytes (tags) with per-dword accumulator origins.
    Frag { bytes: Vec<Tag>, origins: Vec<Option<(u32, u8)>>, addr: Option<i64> },
    Unknown,
}

#[derive(Clone, Debug)]
struct Args {
    m: i64, n: i64, k: i64,
    stride_a: i64, stride_b: i64, stride_c: i64,
    stride_sfa: i64, stride_sfb: i64,
    stride_a_batch: i64, stride_b_batch: i64, stride_c_batch: i64,
    stride_sfa_batch: i64, stride_sfb_batch: i64,
}

impl Args {
    fn small(m: i64, n: i64, k: i64) -> Args {
        Args {
            m, n, k,
            stride_a: k, stride_b: k, stride_c: n,
            stride_sfa: m, stride_sfb: k / 128,
            stride_a_batch: m * k, stride_b_batch: n * k, stride_c_batch: m * n,
            stride_sfa_batch: (k / 128) * m, stride_sfb_batch: (n / 128) * (k / 128),
        }
    }
}

#[derive(Default, Clone)]
pub struct Trace {
    pub contrib: HashMap<(i64, u32, u8), Vec<i64>>,   // lane, accumulator var, dword -> K generations
    pub c_count: HashMap<(i64, i64, i64), usize>,    // m, n, kb
    pub stores: HashSet<i64>,                        // C address (2-byte element index)
    pub a_tags: HashMap<i64, HashSet<(i64, i64)>>,   // kb -> (m, k)
    pub b_tags: HashMap<i64, HashSet<(i64, i64)>>,   // kb -> (n, k)
    pub sa_tags: HashMap<i64, HashSet<(i64, i64)>>,  // kb -> (m, kb)
    pub sb_tags: HashMap<i64, HashSet<(i64, i64)>>,  // kb -> (block, kb)
    pub unknown_ops: HashSet<String>,
    pub buffer_stores: usize,
    pub buffer_store_no_origin: usize,
    pub lds_store_origins: usize,
    pub lds_read_origins: usize,
    pub sample_store: Option<(String, i64, i64)>,
    pub sample_store_origin: usize,
    pub lds_origins_left: usize,
    pub pass1_origin_left: usize,
    pub pass1_lds_store_origins: usize,
    pub pass2_lds_store_origins: usize,
    pub origin_by_addr: HashMap<i64, (u32, u8)>,
    pub tag_by_addr: HashMap<i64, Tag>,
    pub sample_read: Option<(String, i64, i64, usize)>,
    pub sample_epi_read: Option<(String, i64, i64, usize)>,
    pub executed_insts: usize,
    pub mfma_seen: usize,
    pub mfma_no_tags: usize,
    pub dma_writes: usize,
    pub lds_entries: usize,
    pub tag_entries: usize,
    pub tag_a: usize,
    pub tag_b: usize,
    pub tag_sa: usize,
    pub tag_sb: usize,
    pub read_non_u: usize,
    pub mfma_a_frag: usize,
    pub mfma_a_unknown: usize,
    pub lds_miss: usize,
    pub mfma_a_addr_none: usize,
    pub mfma_a_bytes_empty: usize,
    pub sample_operand_miss: Option<(i64, usize)>,
    pub sample_operand_tag: Option<String>,
    pub dup_c: usize,
    pub dup_c_sample: Option<i64>,
    pub sa_mfma: usize,
    pub sb_mfma: usize,
    pub sa_sample: Option<String>,
    pub scale_src: Option<(String, i64)>,
    pub sample_miss: Option<i64>,
    pub sample_hit: Option<i64>,
}

struct Ctx<'a> {
    tid: i64,
    lane: i64,
    wave: i64,
    args: &'a Args,
    vals: HashMap<ValueId, V>,
    vars: HashMap<VarId, V>,
    cur_tile: Option<(String, i64)>,
    lds: Rc<RefCell<HashMap<i64, (Tag, Option<(u32, u8)>)>>>,
    trace: Rc<RefCell<Trace>>,
    unknown_control: usize,
    break_loop: bool,
    stopped: bool,
}

impl<'a> Ctx<'a> {
    fn opnd(&self, o: Opnd) -> V {
        match o {
            Opnd::Val(v) => self.vals.get(&v).cloned().unwrap_or(V::Unknown),
            Opnd::Var(v) => self.vars.get(&v).cloned().unwrap_or(V::Unknown),
            Opnd::I32(x) => V::Int(x as i64),
            Opnd::F32(_) => V::Unknown,
        }
    }
    fn int(&self, o: Opnd) -> Option<i64> {
        match self.opnd(o) { V::Int(x) => Some(x), V::Ptr(_, x) => Some(x), _ => None }
    }
    fn set(&mut self, v: ValueId, val: V) { self.vals.insert(v, val); }
    fn set_var(&mut self, v: VarId, val: V) { self.vars.insert(v, val); }

    fn tag_for(&self, role: PtrRole, off: i64) -> Tag {
        let (m, n, k) = (self.args.m, self.args.n, self.args.k);
        let kb = k / 128;
        match role {
            PtrRole::A => {
                if off >= 0 && off < m * k { Tag::A(off / k, off % k) } else { Tag::U }
            }
            PtrRole::B => {
                // B preshuffled image: (n/16) 16K + (k/16) 256 + (n%16) 16 + k%16
                if off < 0 || off >= n * k { return Tag::U; }
                let n0 = off / (16 * k);
                let rem = off % (16 * k);
                let k0 = rem / 256;
                let rem2 = rem % 256;
                let nn = n0 * 16 + rem2 / 16;
                let kk = k0 * 16 + rem2 % 16;
                Tag::B(nn, kk)
            }
            PtrRole::Sfa => {
                if off >= 0 && off < kb * m { Tag::Sa(off % m, off / m) } else { Tag::U }
            }
            PtrRole::Sfb => {
                if off >= 0 && off < (n / 128) * kb { Tag::Sb(off / kb, off % kb) } else { Tag::U }
            }
            PtrRole::C => Tag::U,
        }
    }
}


fn exec_nodes(nodes: &[Node], lanes: &mut [Ctx<'_>], filter: &[bool]) {
    for n in nodes {
        match n {
            Node::S(st) => {
                for i in 0..lanes.len() {
                    if !filter[i] || lanes[i].stopped { continue; }
                    if exec_stmt(st, &mut lanes[i]) { lanes[i].break_loop = true; }
                }
            }
            Node::If { cond, then, els } => {
                let mut tf = filter.to_vec();
                let mut ef = filter.to_vec();
                for i in 0..lanes.len() {
                    if !filter[i] { tf[i] = false; ef[i] = false; continue; }
                    let c = match lanes[i].opnd(*cond) { V::Int(x) => x != 0, _ => { lanes[i].unknown_control += 1; false } };
                    tf[i] = filter[i] && c;
                    ef[i] = filter[i] && !c;
                }
                exec_nodes(then, lanes, &tf);
                if let Some(els) = els { exec_nodes(els, lanes, &ef); }
            }
            Node::Loop { vars, init, body } => {
                for i in 0..lanes.len() {
                    if !filter[i] { continue; }
                    for (v, o) in vars.iter().zip(init.iter()) { let val = lanes[i].opnd(*o); lanes[i].set_var(*v, val); }
                    lanes[i].stopped = false;
                    lanes[i].break_loop = false;
                }
                for _it in 0..8 {
                    let active = (0..lanes.len()).any(|i| filter[i] && !lanes[i].stopped);
                    if !active { break; }
                    exec_nodes(body, lanes, filter);
                    for i in 0..lanes.len() {
                        if filter[i] && lanes[i].break_loop { lanes[i].break_loop = false; lanes[i].stopped = true; }
                    }
                }
                for i in 0..lanes.len() { if filter[i] { lanes[i].stopped = false; } }
            }
            Node::Region { kind: _, tile, slot: _, version, body } => {
                let mut old: Vec<Option<(String, i64)>> = vec![None; lanes.len()];
                for i in 0..lanes.len() {
                    if !filter[i] { continue; }
                    let ver = lanes[i].int(*version).unwrap_or(-1);
                    old[i] = lanes[i].cur_tile.clone();
                    lanes[i].cur_tile = Some((tile.clone(), ver));
                }
                exec_nodes(body, lanes, filter);
                for i in 0..lanes.len() { if filter[i] { lanes[i].cur_tile = old[i].take(); } }
            }
        }
    }
}

fn exec_stmt(st: &Stmt, cx: &mut Ctx<'_>) -> bool {
    cx.trace.borrow_mut().executed_insts += 1;
    match st {
        Stmt::Let { dst, op } => {
            let val = eval_op(op, cx);
            cx.set(*dst, val);
        }
        Stmt::SetVar { var, value } => { let v = cx.opnd(*value); cx.set_var(*var, v); }
        Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } => {
            let _ = slot;
            let ver = cx.int(*version).unwrap_or(-1);
            cx.cur_tile = Some((tile.clone(), ver));
        }
        Stmt::TileEnd => { /* keep the last (tile, version): MFMAs follow the consumes */ }
        Stmt::Barrier | Stmt::WaitLds { .. } | Stmt::WaitVm { .. } | Stmt::SchedGroup { .. }
        | Stmt::SetPrio { .. } | Stmt::Assume { .. } | Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. } => {}
        Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
            let role = match cx.opnd(*rsrc) { V::Rsrc(r) => r, _ => return false };
            let (so, la) = match (cx.int(*soff), cx.int(*lds_addr)) {
                (Some(v), Some(l)) => (v, l),
                _ => return false,
            };
            let mut src = so + *imm as i64;
            if let Some(v) = voff { if let Some(x) = cx.int(*v) { src += x; } else { return false; } }
            let nbytes = 4 * *dwords as i64;
            if matches!(role, PtrRole::Sfa | PtrRole::Sfb) && cx.trace.borrow().scale_src.is_none() {
                cx.trace.borrow_mut().scale_src = Some((format!("{role:?}"), src));
            }
            let dst = la + *imm as i64 + nbytes * cx.lane;
            for b in 0..nbytes {
                let tag = cx.tag_for(role, src + b);
                cx.lds.borrow_mut().insert(dst + b, (tag.clone(), None));
                cx.trace.borrow_mut().dma_writes += 1;
                match &tag { Tag::A(..) => cx.trace.borrow_mut().tag_a += 1, Tag::B(..) => cx.trace.borrow_mut().tag_b += 1, Tag::Sa(..) => cx.trace.borrow_mut().tag_sa += 1, Tag::Sb(..) => cx.trace.borrow_mut().tag_sb += 1, _ => {} }
                if tag != Tag::U { cx.trace.borrow_mut().tag_by_addr.insert(dst + b, tag); }
            }
        }
        Stmt::StoreLdsB32 { addr, value } => {
            if let (Some(a), V::Word { tags, .. }) = (cx.int(*addr), cx.opnd(*value)) {
                for b in 0..4 { cx.lds.borrow_mut().insert(a + b, (tags.get(b as usize).cloned().unwrap_or(Tag::U), None)); }
            }
        }
        Stmt::StoreLdsB64 { addr, value } | Stmt::StoreLdsB128 { addr, value } => {
            // DMA writes only in pass 1; the epilogue staging writes are also needed in pass 2
            // (they carry accumulator origins), even though pass 2 does not re-run DMA.
            let epi = cx.cur_tile.as_ref().map(|(t, _)| t == "EPI").unwrap_or(false);

            if let (Some(a), V::Frag { bytes, origins, .. }) = (cx.int(*addr), cx.opnd(*value)) {
                cx.trace.borrow_mut().lds_store_origins += origins.iter().filter(|x| x.is_some()).count();
                {
                    if cx.trace.borrow_mut().sample_store.is_none() { cx.trace.borrow_mut().sample_store = Some(("".into(), 0, a)); cx.trace.borrow_mut().sample_store_origin = origins.iter().filter(|x| x.is_some()).count(); }
                    for (b, t) in bytes.iter().enumerate() {
                        let key = a + b as i64;
                        let o = origins.get(b / 4).and_then(|x| *x);
                        if let Some(oo) = o { cx.trace.borrow_mut().origin_by_addr.insert(a + b as i64, oo); }
                        if *t != Tag::U { cx.trace.borrow_mut().tag_by_addr.insert(a + b as i64, t.clone()); }
                        // Do not replace a known accumulator origin by an unknown DMA/epilogue
                        // write at the same byte: the reference is what the store will read.
                        match cx.lds.borrow_mut().get_mut(&key) {
                            Some((_, prev)) if prev.is_some() && o.is_none() => {}
                            Some(slot) => *slot = (t.clone(), o),
                            None => { cx.lds.borrow_mut().insert(key, (t.clone(), o)); }
                        }
                    }
                }
            }
        }
        Stmt::VarMfmaScale16 { var, a, b, scale_a, scale_b, .. } => {
            cx.trace.borrow_mut().mfma_seen += 1;
            let av = cx.opnd(*a); let bv = cx.opnd(*b);
            match &av { V::Frag { bytes, addr, .. } => { cx.trace.borrow_mut().mfma_a_frag += 1; if addr.is_none() { cx.trace.borrow_mut().mfma_a_addr_none += 1; } if bytes.is_empty() { cx.trace.borrow_mut().mfma_a_bytes_empty += 1; } }, V::Unknown => cx.trace.borrow_mut().mfma_a_unknown += 1, _ => {} }
            let lds_snapshot = cx.lds.borrow();
            let tr = cx.trace.borrow();
            // gemm v4 is built with `swap_ab`: the hardware src A slot carries the logical B
            // fragment and src B carries logical A; the scale slots are swapped the same way.
            // Do not assume the slot role: classify every tag by its logical kind.
            let mut a_tags = Vec::new();
            let mut b_tags = Vec::new();
            for tt in frag_tags_fallback(&av, &tr, &lds_snapshot).into_iter()
                .chain(frag_tags_fallback(&bv, &tr, &lds_snapshot)) {
                match tt { Tag::A(..) => a_tags.push(tt), Tag::B(..) => b_tags.push(tt), _ => {} }
            }
            let mut sa_tags = Vec::new();
            let mut sb_tags = Vec::new();
            for tt in word_tags(cx.opnd(*scale_a)).into_iter().chain(word_tags(cx.opnd(*scale_b))) {
                match tt { Tag::Sa(..) => sa_tags.push(tt), Tag::Sb(..) => sb_tags.push(tt), _ => {} }
            }
            drop(tr); drop(lds_snapshot);
            let kb = a_tags.iter().find_map(|t| if let Tag::A(_, k) = t { Some(k / 128) } else { None })
                .or_else(|| b_tags.iter().find_map(|t| if let Tag::B(_, k) = t { Some(k / 128) } else { None }))
                .unwrap_or_else(|| cx.cur_tile.as_ref().map(|x| x.1).unwrap_or(-1));
            if kb < 0 || kb >= cx.args.k / 128 {
                cx.trace.borrow_mut().mfma_no_tags += 1;
                if cx.trace.borrow().sample_operand_miss.is_none() {
                    if let V::Frag { bytes, addr, .. } = &av {
                        let n = bytes.iter().filter(|x| **x != Tag::U).count();
                        cx.trace.borrow_mut().sample_operand_miss = Some((addr.unwrap_or(-1), n));
                        cx.trace.borrow_mut().sample_operand_tag = bytes.iter().find(|x| **x != Tag::U).map(|x| format!("{x:?}"));
                    }
                }
                return false;
            }
            for t in &a_tags { if let Tag::A(m, k) = t { cx.trace.borrow_mut().a_tags.entry(kb).or_default().insert((*m, *k)); } }
            for t in &b_tags { if let Tag::B(n, k) = t { cx.trace.borrow_mut().b_tags.entry(kb).or_default().insert((*n, *k)); } }
            for t in &sa_tags { if let Tag::Sa(m, k) = t { cx.trace.borrow_mut().sa_tags.entry(kb).or_default().insert((*m, *k)); } }
            for t in &sb_tags { if let Tag::Sb(blk, k) = t { cx.trace.borrow_mut().sb_tags.entry(kb).or_default().insert((*blk, *k)); } }
            cx.trace.borrow_mut().sa_mfma += sa_tags.len();
            cx.trace.borrow_mut().sb_mfma += sb_tags.len();
            if cx.trace.borrow().sa_sample.is_none() {
                let s = format!("sa_ops={:?} sb_ops={:?}", word_tags(cx.opnd(*scale_a)), word_tags(cx.opnd(*scale_b)));
                cx.trace.borrow_mut().sa_sample = Some(s);
            }
            for idx in 0..4u8 { cx.trace.borrow_mut().contrib.entry((cx.tid, var.0, idx)).or_default().push(kb); }
            tag_acc_var(cx, *var);
        }
        Stmt::VarMfmaScale { var, a, b, .. } | Stmt::VarMfmaScaleZ { var, a, b, .. } => {
            // 32x32 atoms are not used by v3/v4; keep the tag handling generic.
            let kb = cx.cur_tile.as_ref().map(|x| x.1).unwrap_or(-1);
            for t in frag_tags(cx.opnd(*a)) { if let Tag::A(m, k) = t { cx.trace.borrow_mut().a_tags.entry(kb).or_default().insert((m, k)); } }
            for t in frag_tags(cx.opnd(*b)) { if let Tag::B(n, k) = t { cx.trace.borrow_mut().b_tags.entry(kb).or_default().insert((n, k)); } }
            for idx in 0..4u8 { cx.trace.borrow_mut().contrib.entry((cx.tid, var.0, idx)).or_default().push(kb); }
            tag_acc_var(cx, *var);
        }
        Stmt::BufferStore { rsrc, voff, soff, imm, dwords, value } => {
            let role = match cx.opnd(*rsrc) { V::Rsrc(r) => r, _ => return false };
            if role != PtrRole::C { return false; }
            let so = cx.int(*soff).unwrap_or(0);
            let mut src = so + *imm as i64;
            if let Some(v) = voff { if let Some(x) = cx.int(*v) { src += x; } else { return false; } }
            let frag = cx.opnd(*value);
            let (bytes, origins) = match frag { V::Frag { bytes, origins, .. } => (bytes, origins), _ => return false };
            cx.trace.borrow_mut().buffer_stores += 1;
            if origins.iter().all(|x| x.is_none()) { cx.trace.borrow_mut().buffer_store_no_origin += 1; }
            let nouts = 2 * *dwords as usize;
            for q in 0..nouts {
                let dword = q / 2;
                let from_tag = bytes.get(4 * dword).and_then(|t| if let Tag::Acc(v, i) = t { Some((*v, *i)) } else { None });
                let Some((var, idx)) = origins.get(dword).and_then(|x| *x).or(from_tag) else { continue };
                let kbs = cx.trace.borrow_mut().contrib.get(&(cx.tid, var, idx)).cloned().unwrap_or_default();
                let elem = src / 2 + q as i64;
                let m = elem / cx.args.n;
                let n = elem % cx.args.n;
                if !cx.trace.borrow_mut().stores.insert(elem) { cx.trace.borrow_mut().dup_c += 1; if cx.trace.borrow().dup_c_sample.is_none() { cx.trace.borrow_mut().dup_c_sample = Some(elem); } }
                for kb in kbs { *cx.trace.borrow_mut().c_count.entry((m, n, kb)).or_insert(0) += 1; }
            }
        }
        Stmt::StoreF32x2 { base, byte_off, value } => store_c(cx, *base, *byte_off, *value, 4),
        Stmt::StoreF32x4 { base, byte_off, value } => store_c(cx, *base, *byte_off, *value, 8),
        Stmt::StoreF32 { base, byte_off, value } => store_c(cx, *base, *byte_off, *value, 1),
        Stmt::BreakIf { cond } => {
            return matches!(cx.opnd(*cond), V::Int(x) if x != 0);
        }
        _ => {}
    }
    false
}

fn let_dst_of(_st: &Stmt) -> Option<ValueId> { None }

fn frag_tags(v: V) -> Vec<Tag> {
    match v { V::Frag { bytes, .. } => bytes, _ => Vec::new() }
}
fn frag_tags_fallback(v: &V, trace: &Trace, lds: &HashMap<i64, (Tag, Option<(u32, u8)>)>) -> Vec<Tag> {
    match v {
        V::Frag { bytes, addr, .. } => {
            if !bytes.iter().all(|t| *t == Tag::U) { return bytes.clone(); }
            if let Some(a) = addr {
                let from_lds: Vec<Tag> = (0..32).map(|i| lds.get(&(a + i))
                    .map(|x| x.0.clone())
                    .or_else(|| trace.tag_by_addr.get(&(a + i)).cloned())
                    .unwrap_or(Tag::U)).collect();
                if !from_lds.iter().all(|t| *t == Tag::U) { return from_lds; }
                let out: Vec<Tag> = (0..32).map(|i| trace.tag_by_addr.get(&(a + i)).cloned().unwrap_or(Tag::U)).collect();
                if !out.iter().all(|t| *t == Tag::U) { return out; }
            }
            bytes.clone()
        }
        _ => Vec::new(),
    }
}
fn word_tags(v: V) -> Vec<Tag> {
    match v { V::Word { tags, .. } => tags, _ => Vec::new() }
}

fn store_c(cx: &mut Ctx<'_>, base: Opnd, byte_off: u32, value: Opnd, nouts: usize) {
    let mut off = match cx.opnd(base) { V::Ptr(PtrRole::C, o) => o, _ => return };
    off += byte_off as i64;
    let frag = cx.opnd(value);
    let (bytes, origins) = match frag { V::Frag { bytes, origins, .. } => (bytes, origins), _ => return };
    for q in 0..nouts {
        let dword = q / 2;
        let origin = origins.get(dword).and_then(|x| *x);
        let Some((var, idx)) = origin else { continue };
        let kbs = cx.trace.borrow_mut().contrib.get(&(cx.tid, var, idx)).cloned().unwrap_or_default();
        let elem = (off / 2) as i64 + q as i64;
        let m = elem / cx.args.n;
        let n = elem % cx.args.n;
        if !cx.trace.borrow_mut().stores.insert(elem) { cx.trace.borrow_mut().dup_c += 1; if cx.trace.borrow().dup_c_sample.is_none() { cx.trace.borrow_mut().dup_c_sample = Some(elem); } }
        for kb in kbs {
            *cx.trace.borrow_mut().c_count.entry((m, n, kb)).or_insert(0) += 1;
        }
        let _ = &bytes;
    }
}

fn eval_op(op: &Op, cx: &mut Ctx<'_>) -> V {
    use Op::*;
    let i = |cx: &Ctx<'_>, o: Opnd| cx.int(o);
    match op {
        ConstI32(x) => V::Int(*x as i64),
        KernargI32 { offset } => V::Int(match *offset {
            24 => cx.args.m, 28 => cx.args.n, 32 => cx.args.k, 36 => 1,
            40 => cx.args.stride_a, 44 => cx.args.stride_b, 48 => cx.args.stride_c,
            52 => cx.args.stride_a_batch, 56 => cx.args.stride_b_batch, 60 => cx.args.stride_c_batch,
            80 => cx.args.stride_sfa, 84 => cx.args.stride_sfb,
            88 => cx.args.stride_sfa_batch, 92 => cx.args.stride_sfb_batch,
            _ => 0,
        }),
        KernargPtr { offset } => V::Ptr(match *offset { 0 => PtrRole::A, 8 => PtrRole::B, 16 => PtrRole::C, 64 => PtrRole::Sfa, 72 => PtrRole::Sfb, _ => PtrRole::C }, 0),
        PtrAdd { base, byte_off } => match (cx.opnd(*base), i(cx, *byte_off)) {
            (V::Ptr(r, b), Some(o)) => V::Ptr(r, b + o),
            _ => V::Unknown,
        },
        BufferRsrcDyn { base, .. } => match cx.opnd(*base) { V::Ptr(r, _) => V::Rsrc(r), _ => V::Unknown },
        ReadFirstLane(x) => cx.opnd(*x),
        WorkitemId => V::Int(cx.tid),
        WorkgroupIdX | WorkgroupIdY | WorkgroupIdZ => V::Int(0),
        AddI32(a, b) => match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x.wrapping_add(y)), _ => V::Unknown },
        MulI32(a, b) => match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x.wrapping_mul(y)), _ => V::Unknown },
        ShlI32(a, b) => {
            if let V::Word { tags, origin } = cx.opnd(*a) { return V::Word { tags, origin }; }
            match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x.wrapping_shl(y as u32)), _ => V::Unknown }
        }
        LshrI32(a, b) => {
            if let V::Word { tags, origin } = cx.opnd(*a) { return V::Word { tags, origin }; }
            match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(((x as u32) >> (y as u32)) as i64), _ => V::Unknown }
        }
        AndI32(a, b) => {
            if let V::Word { tags, origin } = cx.opnd(*a) { return V::Word { tags, origin }; }
            match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x & y), _ => V::Unknown }
        }
        XorI32(a, b) => {
            if let V::Word { tags, origin } = cx.opnd(*a) { return V::Word { tags, origin }; }
            match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x ^ y), _ => V::Unknown }
        }
        MinI32(a, b) => match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int(x.min(y)), _ => V::Unknown },
        CmpLtI32(a, b) => match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int((x < y) as i64), _ => V::Unknown },
        CmpGeI32(a, b) => match (i(cx, *a), i(cx, *b)) { (Some(x), Some(y)) => V::Int((x >= y) as i64), _ => V::Unknown },
        NotI1(a) => match i(cx, *a) { Some(x) => V::Int((x == 0) as i64), None => V::Unknown },
        Select { cond, a, b, .. } => match i(cx, *cond) { Some(0) => cx.opnd(*b), Some(_) => cx.opnd(*a), None => V::Unknown },
        BitcastI32(a) | BitcastF32(a) => cx.opnd(*a),
        CvtPkBf16F32(a, b) => {
            let mut t = word_tags(cx.opnd(*a));
            t.extend(word_tags(cx.opnd(*b)));
            V::Word { tags: t, origin: word_origin(cx.opnd(*a)).or(word_origin(cx.opnd(*b))) }
        }
        CvtPkFp8F32(a, b) => { let mut t = word_tags(cx.opnd(*a)); t.extend(word_tags(cx.opnd(*b))); V::Word { tags: t, origin: None } }
        Frag2(a, b) => {
            let mut bytes = word_tags(cx.opnd(*a)); bytes.extend(word_tags(cx.opnd(*b)));
            V::Frag { bytes, origins: vec![word_origin(cx.opnd(*a)), word_origin(cx.opnd(*b))], addr: None }
        }
        Frag4(a, b, c, d) => {
            let mut bytes = Vec::new(); let mut origins = Vec::new();
            for x in [a, b, c, d] { bytes.extend(word_tags(cx.opnd(*x))); origins.push(word_origin(cx.opnd(*x))); }
            V::Frag { bytes, origins, addr: None }
        }
        Frag2Lane { frag, index } => {
            let v = cx.opnd(*frag);
            let var_origin = match frag { Opnd::Var(v) => Some((v.0, *index)), _ => None };
            match v { V::Frag { bytes, origins, .. } => {
                let d = (*index as usize).min(origins.len().saturating_sub(1));
                let tags = bytes.iter().skip(4 * *index as usize).take(4).cloned().collect();
                V::Word { tags, origin: var_origin.or(origins.get(d).and_then(|x| *x)) }
            }, _ => V::Unknown }
        }
        Frag4Lane { frag, index } => {
            let v = cx.opnd(*frag);
            let var_origin = match frag { Opnd::Var(v) => Some((v.0, *index)), _ => None };
            match v { V::Frag { bytes, origins, .. } => {
                let tags = bytes.iter().skip(4 * *index as usize).take(4).cloned().collect();
                V::Word { tags, origin: var_origin.or(origins.get(*index as usize).and_then(|x| *x)) }
            }, _ => V::Unknown }
        }
        FragZero { .. } => V::Frag { bytes: Vec::new(), origins: vec![None; 4], addr: None },
        DppQuadPerm { x, .. } | PermB32(x, ..) => {
            let v = cx.opnd(*x);
            match v { V::Word { tags, origin } => V::Word { tags, origin }, _ => V::Unknown }
        }
        LoadLdsB128 { addr } => lds_read(cx, *addr, 16),
        LoadLdsB128x2 { lo, hi } => {
            let (Some(l), Some(h)) = (cx.int(*lo), cx.int(*hi)) else { return V::Unknown };
            let mut bytes = Vec::new(); let mut origins = Vec::new();
            for a in [l, h] { let v = lds_read(cx, Opnd::I32(a as i32), 16); if let V::Frag { bytes: b, origins: o, .. } = v { bytes.extend(b); origins.extend(o); } }
            V::Frag { bytes, origins, addr: Some(l) }
        }
        LoadLdsB64 { addr } => lds_read(cx, *addr, 8),
        LoadLdsB32 { addr } => lds_read_word(cx, *addr),
        BufferLoad { rsrc, voff, soff, imm, dwords } => {
            let role = match cx.opnd(*rsrc) { V::Rsrc(r) => r, _ => return V::Unknown };
            let so = cx.int(*soff).unwrap_or(0);
            let mut src = so + *imm as i64;
            if let Some(v) = voff { if let Some(x) = cx.int(*v) { src += x; } else { return V::Unknown; } }
            let tags: Vec<Tag> = (0..4 * *dwords as i64).map(|b| cx.tag_for(role, src + b)).collect();
            if *dwords == 1 { V::Word { tags, origin: None } }
            else { V::Frag { bytes: tags, origins: vec![None; *dwords as usize], addr: None } }
        }
        _ => { cx.trace.borrow_mut().unknown_ops.insert(format!("{op:?}").split('(').next().unwrap_or("op").to_string()); V::Unknown }
    }
}

fn tag_acc_var(cx: &mut Ctx<'_>, var: VarId) {
    let mut f = match cx.vars.get(&var).cloned() { Some(V::Frag { bytes, origins, .. }) => (bytes, origins), _ => (vec![Tag::U; 16], vec![None; 4]) };
    while f.0.len() < 16 { f.0.push(Tag::U); }
    while f.1.len() < 4 { f.1.push(None); }
    for idx in 0..4u8 {
        for b in 0..4 { f.0[4 * idx as usize + b] = Tag::Acc(var.0, idx); }
        f.1[idx as usize] = Some((var.0, idx));
    }
    cx.vars.insert(var, V::Frag { bytes: f.0, origins: f.1, addr: None });
}

fn word_origin(v: V) -> Option<(u32, u8)> { match v { V::Word { origin, .. } => origin, _ => None } }

fn lds_read(cx: &mut Ctx<'_>, addr: Opnd, n: usize) -> V {
    let Some(a) = cx.int(addr) else { return V::Unknown };
    let (tile, ver) = cx.cur_tile.clone().unwrap_or_else(|| (String::new(), 0));
    let mut bytes = Vec::new();
    for i in 0..n {
        let exact = cx.lds.borrow().get(&(a + i as i64)).map(|x| x.0.clone());
        let mut tag = exact.clone().unwrap_or(Tag::U);
        if exact.is_none() && cx.trace.borrow().lds_miss < 16 { cx.trace.borrow_mut().sample_miss = Some(a + i as i64); }
        if exact.is_some() && cx.trace.borrow().sample_hit.is_none() { cx.trace.borrow_mut().sample_hit = Some(a + i as i64); }
        if exact.is_none() { cx.trace.borrow_mut().lds_miss += 1; }
        if tag == Tag::U { if let Some(t) = cx.trace.borrow().tag_by_addr.get(&(a + i as i64)) { tag = t.clone(); } }
        if tag != Tag::U { cx.trace.borrow_mut().read_non_u += 1; }
        bytes.push(tag);
    }
    let mut origins = vec![None; n / 4];
    for d in 0..n / 4 {
        let addr = a + 4 * d as i64;
        if let Some((_, o)) = cx.lds.borrow_mut().get(&addr) { origins[d] = *o; }
        if origins[d].is_none() {
            // LDS epilogue aliases the dead ring/panel storage; the same byte may have been
            // written under another (tile, generation) key. Use an existing accumulator
            // provenance at this byte address as the origin fallback.
            if let Some(o) = cx.trace.borrow_mut().origin_by_addr.get(&addr) { origins[d] = Some(*o); }
        }
    }
    let n_o = origins.iter().filter(|x| x.is_some()).count();
    cx.trace.borrow_mut().lds_read_origins += n_o;
    if cx.trace.borrow_mut().sample_read.is_none() { cx.trace.borrow_mut().sample_read = Some((tile.clone(), ver, a, n_o)); }
    if tile == "EPI" && cx.trace.borrow_mut().sample_epi_read.is_none() { cx.trace.borrow_mut().sample_epi_read = Some((tile.clone(), ver, a, n_o)); }
    V::Frag { bytes, origins, addr: Some(a) }
}
fn lds_read_word(cx: &mut Ctx<'_>, addr: Opnd) -> V {
    let v = lds_read(cx, addr, 4);
    match v { V::Frag { bytes, .. } => V::Word { tags: bytes, origin: None }, _ => V::Unknown }
}

fn run_all(f: &Function, args: &Args) -> (Trace, usize) {
    let lds: Rc<RefCell<HashMap<i64, (Tag, Option<(u32, u8)>)>>> = Rc::new(RefCell::new(HashMap::new()));
    let trace: Rc<RefCell<Trace>> = Rc::new(RefCell::new(Trace::default()));
    let mut lanes: Vec<Ctx<'_>> = (0..256i64).map(|tid| Ctx {
        tid,
        lane: tid % 64,
        wave: tid / 64,
        args,
        vals: HashMap::new(),
        vars: HashMap::new(),
        cur_tile: None,
        lds: Rc::clone(&lds),
        trace: Rc::clone(&trace),
        unknown_control: 0,
        break_loop: false,
        stopped: false,
    }).collect();
    let filter = vec![true; lanes.len()];
    exec_nodes(f.body(), &mut lanes, &filter);
    let unknown = lanes.iter().map(|c| c.unknown_control).sum();
    let mut t = trace.borrow().clone();
    let lg = lds.borrow();
    t.lds_entries = lg.len();
    t.tag_a = lg.values().filter(|x| matches!(x.0, Tag::A(..))).count();
    t.tag_b = lg.values().filter(|x| matches!(x.0, Tag::B(..))).count();
    t.tag_sa = lg.values().filter(|x| matches!(x.0, Tag::Sa(..))).count();
    t.tag_sb = lg.values().filter(|x| matches!(x.0, Tag::Sb(..))).count();
    (t, unknown)
}

pub fn prove_ir(f: &Function, cfg: &GemmConfig, m: i64, n: i64, k: i64) -> Result<Trace, ProofError> {
    if cfg.cta_m != 256 || cfg.cta_n != 256 { return Err(ProofError("E3 expects the 256x256 CTA".into())); }
    if k % 128 != 0 { return Err(ProofError("K must be a multiple of 128".into())); }
    let args = Args::small(m, n, k);
    let (trace, _unknown_control) = run_all(f, &args);
    if !trace.unknown_ops.is_empty() {
        return Err(ProofError(format!("unsupported/unknown IR during symbolic execution: {:?}", trace.unknown_ops)));
    }
    Ok(trace)
}

/// Check the E3 theorem on the trace. `expected_drop` is the K generation the kernel proof hook
/// removed (None for the correct kernel).
pub fn check_trace(trace: &Trace, cfg: &GemmConfig, m: i64, n: i64, k: i64, expected_drop: Option<i64>) -> Result<(), String> {
    let kb_n = k / 128;
    if trace.stores.is_empty() { return Err("no C stores executed (symbolic execution did not reach the epilogue)".into()); }
    for mm in 0..m {
        for nn in 0..n {
            for kb in 0..kb_n {
                if Some(kb) == expected_drop { continue; }
                let c = trace.c_count.get(&(mm, nn, kb)).copied().unwrap_or(0);
                if c != 1 {
                    return Err(format!("missing/duplicate contribution C[{mm},{nn}] K block {kb} (k={}): {c} times", kb * 128));
                }
            }
        }
    }
    // operand coverage: for every K block present, all A rows/K and B columns/K must have been seen
    for kb in 0..kb_n {
        if Some(kb) == expected_drop { continue; }
        let a = trace.a_tags.get(&kb).cloned().unwrap_or_default();
        let b = trace.b_tags.get(&kb).cloned().unwrap_or_default();
        for mm in 0..cfg.cta_m {
            for kk in kb * 128..(kb + 1) * 128 {
                if !a.contains(&(mm, kk)) { return Err(format!("A operand tag missing A[{mm},{kk}] for K block {kb}")); }
            }
        }
        for nn in 0..cfg.cta_n {
            for kk in kb * 128..(kb + 1) * 128 {
                if !b.contains(&(nn, kk)) { return Err(format!("B operand tag missing B[{nn},{kk}] for K block {kb}")); }
            }
        }
        for mm in 0..cfg.cta_m {
            if !trace.sa_tags.get(&kb).map_or(false, |s| s.iter().any(|&(x, _)| x == mm)) {
                return Err(format!("A scale tag missing sa[{mm},{kb}]"));
            }
        }
        for blk in 0..cfg.cta_n / 128 {
            if !trace.sb_tags.get(&kb).map_or(false, |s| s.iter().any(|&(x, _)| x == blk)) {
                return Err(format!("B scale tag missing sb[{blk},{kb}]"));
            }
        }
    }
    Ok(())
}
