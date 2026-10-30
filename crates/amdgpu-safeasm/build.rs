//! Generates the typed opcode templates (`generated_ops.rs`) from `opspec.tbl` and validates every row against the backend table.
use std::fmt::Write;

fn main() {
    println!("cargo:rerun-if-changed=opspec.tbl");
    let spec = std::fs::read_to_string("opspec.tbl").expect("opspec.tbl");
    let tbl_path = "../amdgpu-compiler/src/backend/llvm24.tbl";
    println!("cargo:rerun-if-changed={tbl_path}");
    // key -> arity (number of source classes) from the backend table; absent when building outside the workspace
    let mut arity: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if let Ok(t) = std::fs::read_to_string(tbl_path) {
        for l in t.lines() {
            if l.starts_with('#') || l.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = l.split('|').collect();
            if parts.len() < 3 {
                continue;
            }
            let key = parts[0].trim().split(':').next().unwrap().to_string();
            let sig = parts[1].trim();
            let srcs = sig.split("<-").nth(1).unwrap_or("").split_whitespace().count();
            arity.entry(key).or_insert(srcs);
        }
    }
    let mut out = String::new();
    let mut cov = String::from("pub const TYPED_OPS: &[(&str, &str, &str)] = &[\n");
    for (n, l) in spec.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = l.split('|').map(|s| s.trim()).collect();
        assert!(f.len() == 3, "opspec.tbl:{}: expected `method | key | form`", n + 1);
        let (m, k, form) = (f[0], f[1], f[2]);
        let want = match form {
            "vop1" => 1,
            "vop2" | "vop2_imm0" | "vop3_2" | "cmp" | "store_vm" | "store_vm2" | "store_vm4" => 2,
            "vop3_3" => 3,
            other => panic!("opspec.tbl:{}: unknown form `{other}`", n + 1),
        };
        if !arity.is_empty() {
            let got = arity.get(k).unwrap_or_else(|| panic!("opspec.tbl:{}: `{k}` is not in llvm24.tbl", n + 1));
            assert!(*got == want, "opspec.tbl:{}: `{k}` has {got} sources in llvm24.tbl, form `{form}` needs {want}", n + 1);
        }
        let _ = writeln!(cov, "    (\"{m}\", \"{k}\", \"{form}\"),");
        let hz = "self.note_valu_write(dst.dst_id());";
        match form {
            "vop1" => {
                let _ = write!(out, "    /// `{m} dst, src0` (generated from opspec.tbl: `{k}`).\n    pub fn {m}<D: Dst<1>, A: Operand>(&mut self, dst: D, a: A) {{\n        self.push(inst(\"{k}\", Some(dst.dst_id()), vec![a.to_src()]));\n        {hz}\n    }}\n\n");
            }
            "vop2" => {
                let _ = write!(out, "    /// `{m}_e32 dst, src0, vsrc1`: src0 any operand, src1 a VGPR (generated from opspec.tbl: `{k}`).\n    pub fn {m}<D: Dst<1>, A: Operand, B: VgprOnly>(&mut self, dst: D, a: A, b: B) {{\n        self.push(inst(\"{k}\", Some(dst.dst_id()), vec![a.to_src(), b.to_src()]));\n        {hz}\n    }}\n\n");
            }
            "vop2_imm0" => {
                let _ = write!(out, "    /// `{m} dst, imm, vsrc`: the shift amount is a compile-time inline constant (generated from opspec.tbl: `{k}`).\n    pub fn {m}<D: Dst<1>, const K: i32, B: VgprOnly>(&mut self, dst: D, _k: Imm<K>, b: B) {{\n        let () = Imm::<K>::OK;\n        self.push(inst(\"{k}\", Some(dst.dst_id()), vec![ASrc::Imm(K as i64), b.to_src()]));\n        {hz}\n    }}\n\n");
            }
            "vop3_2" => {
                let _ = write!(out, "    /// `{m} dst, a, b` (VOP3: at most one scalar source, no literal; generated from opspec.tbl: `{k}`).\n    pub fn {m}<D: Dst<1>, A: Operand, B: Operand>(&mut self, dst: D, a: A, b: B) {{\n        let () = Bus1::<A, B>::OK;\n        self.push(inst(\"{k}\", Some(dst.dst_id()), vec![a.to_src(), b.to_src()]));\n        {hz}\n    }}\n\n");
            }
            "vop3_3" => {
                let _ = write!(out, "    /// `{m} dst, a, b, c` (VOP3: at most one scalar source, no literal; generated from opspec.tbl: `{k}`).\n    pub fn {m}<D: Dst<1>, A: Operand, B: Operand, C: Operand>(&mut self, dst: D, a: A, b: B, c: C) {{\n        let () = Bus3::<A, B, C>::OK;\n        self.push(inst(\"{k}\", Some(dst.dst_id()), vec![a.to_src(), b.to_src(), c.to_src()]));\n        {hz}\n    }}\n\n");
            }
            "cmp" => {
                let _ = write!(out, "    /// `{m}_e32 vcc, src0, vsrc1`: `Vcc<Free>` in, `Vcc<Live>` out (generated from opspec.tbl: `{k}`).\n    pub fn {m}<A: Operand, B: VgprOnly>(&mut self, _vcc: Vcc<Free>, a: A, b: B) -> Vcc<Live> {{\n        self.push(inst(\"{k}\", None, vec![a.to_src(), b.to_src()]));\n        self.vcc_written();\n        Vcc {{ _s: PhantomData }}\n    }}\n\n");
            }
            "store_vm" => {
                let _ = write!(out, "    /// `{m} vaddr[2], vdata offset:N`: occupies a vmcnt slot, no value comes back (generated from opspec.tbl: `{k}`).\n    pub fn {m}(&mut self, addr: &impl Readable<2>, data: &impl Readable<1>, offset: i32) {{\n        self.push(AInst {{ op: \"{k}\", dst: None, srcs: vec![ASrc::Reg(addr.reg_id()), ASrc::Reg(data.reg_id())], imm: offset as i64 }});\n        self.issue_store(Queue::Vm);\n    }}\n\n");
            }
            "store_vm2" | "store_vm4" => {
                let w = if form == "store_vm2" { 2 } else { 4 };
                let _ = write!(out, "    /// `{m} vaddr[2], vdata[{w}] offset:N`: occupies a vmcnt slot, no value comes back (generated from opspec.tbl: `{k}`).\n    pub fn {m}(&mut self, addr: &impl Readable<2>, data: &impl Readable<{w}>, offset: i32) {{\n        self.push(AInst {{ op: \"{k}\", dst: None, srcs: vec![ASrc::Reg(addr.reg_id()), ASrc::Reg(data.reg_id())], imm: offset as i64 }});\n        self.issue_store(Queue::Vm);\n    }}\n\n");
            }
            _ => unreachable!(),
        }
    }
    cov.push_str("];\n");
    let dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{dir}/generated_ops.rs"), format!("impl Seq {{\n{out}}}\n\n{cov}")).unwrap();
}