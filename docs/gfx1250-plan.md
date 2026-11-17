# gfx1250 (MI450) code generation plan

Staged, each stage one branch gated like every other change (gfx950 md5-identical: the asm, `.co`, added-table and certificate manifests;
`run_all`). ISA facts come only from the public CDNA5 ISA reference and public LLVM (docs/target-gfx1250.md); safety rules that keep the
hardware from hanging are verifier / compile errors stated generically.

| Stage | Content | State |
|---|---|---|
| G0 | target plumbing: `amdgpu_target::gfx1250::GFX1250` (the whole `Target`), `amdgpu_target::by_name`, `target::cur()` / `with_target` (the compilation's target, gfx950 by default), `compile_target`, `amdc emit --target gfx1250`; the toolchain pin `backend.gfx1250.lock` (the MI450 nodes' ROCm 10.2.0 LLVM, llc built against its libLLVM) | done |
| G1 | wave32 MIR ops (VALU / SALU / SMEM / global / DS, split waits, cache policy th / scope / nv, WMMA), per-target `op_attrs`, the gfx1250 emitter (kernel descriptor: VGPR blocks of 16, SGPR field 0, LDS in 2048-byte units, code object v5) and in-process encoder with `llvm-mc -mcpu=gfx1250 -show-encoding` as the byte oracle, the LLVM tail table for `llc -mcpu=gfx1250`; the TDM statements (`TensorLoadToLds` / `TensorStoreFromLds` with a raw-words form for exact replays); first kernels (elementwise, rmsnorm, GEMV, a WMMA probe); VGPRs v0..v255 only | next |
| G2 | waits per split counter (per-class order, mixed classes wait to 0, TENSOR / ASYNC always explicit), the gfx1250 hazard rows (`s_wait_alu`, WMMA / TRANS co-execution, ...), VGPR MSB windowing (`s_set_vgpr_msb`), the scheduler's gfx1250 machine model | |
| G3 | TDM (typed descriptor built in SGPRs), async LDS, cluster loads, split / named / cluster barriers in the DSL / IR; the Weaver: TDM / async producers as ring stages, barrier signal early / wait late; the protocol verifier's TDM / async axes | |
| G4 | hardware validation on MI450 (probes, op checks, perf) | |
| G5 | hang-safety verifier: barrier balance, cluster-only multicast fields, descriptor validity, bounded outstanding tensor / async operations, completion before reads, no unbounded sleeps; one mutation test per rule | |
