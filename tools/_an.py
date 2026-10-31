import json, collections
C = collections.Counter
d = json.load(open(r"tools/llvm_feature_audit.json"))
T = d["tests"]; U = set(d["universe"]["universe"]); s = d["summary"]
print("N", len(T), dict(C(t["ext"] for t in T)), dict(C(t["dir"] for t in T)))
print("kind", dict(C(t["kind"] for t in T)), "status", dict(C(t["status"] for t in T)))
print("ext x kind", {e: dict(C(t["kind"] for t in T if t["ext"] == e)) for e in ("ll", "mir")})
cg = [t for t in T if t["kind"] == "codegen"]
print("codegen by ext", dict(C(t["ext"] for t in cg)))
G = ("gfx950", "gfx94x")
g = [t for t in cg if any(c in G for c in t["target_classes"])]
print("gfx950-class codegen", len(g), dict(C(t["status"] for t in g)))
print("gfx950 only", len([t for t in cg if "gfx950" in t["target_classes"]]), "gfx94x only", len([t for t in cg if "gfx94x" in t["target_classes"] and "gfx950" not in t["target_classes"]]))
print("files with gfx950 anywhere in classes (any kind)", len([t for t in T if "gfx950" in t["target_classes"]]))
# class breakdown for codegen tests by 'has class'
for cl in ("gfx950", "gfx94x", "gfx9x", "gfx10", "gfx11", "gfx12", "pre-gfx9", "r600", "unknown"):
    print(" cls", cl, len([t for t in cg if cl in t["target_classes"]]), "any-kind", len([t for t in T if cl in t["target_classes"]]))
cov = [t for t in g if t["status"] == "COVERED"]
print("COVERED", len(cov), "vacuous", sum(1 for t in cov if t.get("vacuous")))
for t in cov: print("  C", t["path"], t["target_classes"], len(t.get("opcodes_gfx950_class", [])), "vac" if t.get("vacuous") else "")
par = [t for t in g if t["status"] == "PARTIAL"]
print("PARTIAL", len(par), "with missing opcodes", sum(1 for t in par if t["missing_opcodes"]), "only feature", sum(1 for t in par if not t["missing_opcodes"]))
print("gfx950-class OOS reasons", C(t["reasons"][0] for t in g if t["status"] == "OUT-OF-SCOPE").most_common(12))
hardf = C()
for t in g:
    if t["status"] == "OUT-OF-SCOPE":
        for f in t["features"]:
            if f in ("call_nonintrinsic", "indirect_call", "exceptions", "alloca", "dynamic_alloca", "stacksave", "debug_info", "varargs", "non_kernel_function", "stack_objects", "private_memory"):
                hardf[f] += 1
print("gfx950-class OOS hard feature occurrences", hardf.most_common())
print("TOP30 missing gfx950-class")
for r in s["top_missing_opcodes_gfx950_class"][:30]: print(" ", r["opcode"], r["tests"], r["examples"][:3])
print("TOP30 missing all codegen")
for r in s["top_missing_opcodes_all_codegen"][:30]: print(" ", r["opcode"], r["tests"])
# relaxed: gfx9 family ISA (gfx9x+94x+950) ignoring target gate
G9 = ("gfx9x", "gfx94x", "gfx950")
HARD = ("call_nonintrinsic", "indirect_call", "exceptions", "alloca", "dynamic_alloca", "stacksave", "debug_info", "varargs", "non_kernel_function", "stack_objects", "private_memory")
g9 = [t for t in cg if any(c in G9 for c in t["target_classes"])]
def ops9(t):
    o = set()
    for c in G9: o |= set(t["opcodes_by_class"].get(c, []))
    return o
relaxed = C()
for t in g9:
    o = ops9(t); hard = [f for f in t["features"] if f in HARD]
    relaxed["hard-OOS" if hard else ("covered" if not (o - U) else "partial")] += 1
print("gfx9-family codegen (relaxed)", len(g9), dict(relaxed))
print("features all .ll", C(f for t in T if t["ext"] == "ll" for f in t["features"]).most_common(30))
print("features gfx950-class", C(f for t in g for f in t["features"]).most_common(30))
print("cc", s["calling_conventions"][:12]); print("attrs", s["attributes"][:25]); print("aspaces", s["address_spaces"]); print("types", s["types"]); print("mattr", s["mattr"][:15])
print("distinct amdgcn intrinsics", s["n_distinct_amdgcn_intrinsics"], "gfx950-class distinct", len(set(i for t in g for i in t["amdgcn_intrinsics"])))
print("amdgcn intr top", s["intrinsics_amdgcn_top"][:15])
print("requires", s["tests_with_requires"], "unsupported", s["tests_with_unsupported"], "xfail", s["tests_with_xfail"])
print("kernel-style: tests with amdgpu_kernel cc", sum(1 for t in T if "amdgpu_kernel" in t["calling_conventions"]), "with non-kernel func", sum(1 for t in T if "non_kernel_function" in t["features"]), "with call", sum(1 for t in T if "call_nonintrinsic" in t["features"]), "alloca", sum(1 for t in T if "alloca" in t["features"]), "atomics", sum(1 for t in T if any(f in t["features"] for f in ("atomicrmw","cmpxchg","fence","atomic_load","atomic_store"))), "debug", sum(1 for t in T if "debug_info" in t["features"]), "exceptions", sum(1 for t in T if "exceptions" in t["features"]), "varargs", sum(1 for t in T if "varargs" in t["features"]), "lds", sum(1 for t in T if "lds" in t["features"]), "barrier", sum(1 for t in T if "barrier" in t["features"]), "kernarg", sum(1 for t in T if "kernarg" in t["features"]), "workgroup_id", sum(1 for t in T if "workgroup_id" in t["features"]))
print("universe", d["universe"]["n_mops"], len(U), "mops with mnemonics", len(d["universe"]["mop_to_mnemonics"]))
print("distinct opcodes in tests", s["n_distinct_opcodes_in_tests"], "we have", s["opcodes_in_tests_that_we_have"])
print("top opcodes", s["top_opcodes_in_tests"][:20])
print("codegen with opcode checks", s["codegen_tests_with_opcode_checks"], "all-in-universe(any target)", s["codegen_tests_all_opcodes_in_our_universe(any_target)"])
print("mfma tests gfx950-class", [t["path"] for t in g if any(o.startswith("v_mfma") or o.startswith("v_smfmac") for o in t["opcodes"])][:20])
print("universe sample", sorted(U)[:200])