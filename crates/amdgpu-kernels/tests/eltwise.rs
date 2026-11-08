//! Batch 2 streaming elementwise kernels (docs/op-coverage-membound.md): the name grammar, strict GPU-semantics verification (buffer accesses hardware-guarded, no errors) and assembly.
use amdgpu_compiler::gpu_verify::analyse;
use amdgpu_kernels::kernel_by_name;

const KERNELS: &[&str] = &["silu_mul_bf16_d4096", "silu_mul_bf16_d1024", "silu_mul_bf16_d24", "silu_mul_bf16_d14336", "gelu_tanh_mul_bf16_d4096", "gelu_tanh_mul_bf16_d11008", "add_bf16"];

#[test]
fn the_eltwise_kernels_verify_strict_and_compile() {
    for k in KERNELS {
        let f = kernel_by_name(k).unwrap_or_else(|| panic!("{k} is registered"));
        let r = analyse(&f).unwrap();
        assert!(r.errors().is_empty(), "{k}: {}", r.render());
        let c = amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.vgpr <= 64, "{k}: a streaming kernel stays small (occupancy): {} VGPR", c.vgpr);
        assert!(c.asm.contains("buffer_load_dwordx4") && c.asm.contains("buffer_store_dwordx4"), "{k}: one b128 per lane");
    }
}

#[test]
fn the_name_grammar_rejects_what_the_kernels_do_not_support() {
    for bad in ["silu_mul_bf16_d0", "silu_mul_bf16_d12", "silu_mul_bf16_d70000", "silu_mul_bf16_dx", "gelu_tanh_mul_bf16_d7", "add_bf16x"] {
        assert!(kernel_by_name(bad).is_none(), "{bad}");
    }
}

#[test]
fn the_block_shape_follows_d() {
    // threads = min(4, ceil(D / 8 / 64)) * 64: a short row gets a single wave, a long row 256 threads and several column tiles
    for (k, threads) in [("silu_mul_bf16_d24", 64), ("silu_mul_bf16_d1024", 128), ("silu_mul_bf16_d4096", 256), ("silu_mul_bf16_d14336", 256)] {
        assert_eq!(kernel_by_name(k).unwrap().threads, threads, "{k}");
    }
}