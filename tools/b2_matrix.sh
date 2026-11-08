#!/usr/bin/env bash
# Batch-2 matrix: emit + assemble every kernel, then correctness (small odd shapes) and bandwidth (~1 GB working sets) with the host drivers in tests/gpu. One line per case.
# usage (on a gfx950 node, from the repo root, with $AMDC = a built amdc): bash tools/b2_matrix.sh [out]   (the vendor harness is separate: tools/vendor_membound.sh)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4}
A=${AMDC:-/tmp/target-b2/debug/amdc}; R=${ROCM:-/shared/apps/ubuntu/opt/rocm-7.0.2}; CL=${CLANG:-/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang}; O=${1:-build/b2}
mkdir -p $O
for h in eltwise rows quant cast rope cache topk dequant; do [ -x $O/${h}_host ] || $R/bin/amdclang++ -x hip -O2 --offload-arch=gfx950 --rocm-path=$R -std=c++17 tests/gpu/${h}_host.cpp -o $O/${h}_host -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib 2>&1 | grep error; done
mk() { [ -f $O/$1.co ] && [ $O/$1.co -nt $A ] && return 0; $A emit $1 --backend=own --out $O/$1.s > $O/$1.log 2>&1 || { echo "EMIT FAIL $1"; head -4 $O/$1.log; return 1; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o $O/$1.co $O/$1.s; }
one() { lab=$1; shift; out=$("$@" 2>&1); echo "$lab | $(echo "$out" | grep -E '^check' | sed -E 's/.*(bad[a-z_]*=[0-9]+).*(OK|FAIL)$/\1 \2/' | head -1) | $(echo "$out" | grep -E '^timing' | sed -E 's/.*ms\/iter[^,]*, ([0-9.]+) GB\/s.*/\1 GB\/s/; s/^timing: ([0-9.]+) ms\/iter \(([0-9.]+) us\).*/\2 us/' | head -1)"; }
DQ=$O/dequant_host; E=$O/eltwise_host; RW=$O/rows_host; Q=$O/quant_host; C=$O/cast_host; RP=$O/rope_host; CA=$O/cache_host; TK=$O/topk_host
for k in silu_mul_bf16_d4096 silu_mul_bf16_d1024 silu_mul_bf16_d14336 silu_mul_bf16_d24 gelu_tanh_mul_bf16_d4096 add_bf16 cvt_f32_bf16 cvt_bf16_f32 quant_bf16_fp8_static rmsnorm_bf16_n4096 rmsnorm_bf16_n1000 rmsnorm_bf16_n200 add_rmsnorm_bf16_n4096 layernorm_bf16_n4096 layernorm_bf16_n776 softmax_bf16_n4096 softmax_bf16_n1000 softmax_bf16_n8192 quant_token_bf16_fp8_n4096 quant_token_bf16_fp8_n1000 quant_group128_bf16_fp8_n4096 quant_group128_bf16_fp8_n640 quant_token_bf16_int8_n4096 quant_token_bf16_int8_n1000 dequant_fp8_bf16_static dequant_fp8_bf16_token_n4096 dequant_fp8_bf16_group128_n4096 dequant_fp8_bf16_token_n1000 silu_mul_quant_fp8_d4096 silu_mul_quant_fp8_d1000 rope_neox_bf16_h32_d128 rope_gptj_bf16_h32_d128 rope_neox_bf16_h8_d128 rope_neox_bf16_h40_d64 reshape_cache_flash_bf16_h8_d128 reshape_cache_flash_bf16_h5_d64 topk_softmax_bf16_e128_k8 topk_softmax_renorm_bf16_e256_k8; do mk $k; done
one "silu_mul d4096 corr+perf" $E $O/silu_mul_bf16_d4096.co silu_mul_bf16_d4096 silu_mul 43690 4096 50
one "silu_mul d1024" $E $O/silu_mul_bf16_d1024.co silu_mul_bf16_d1024 silu_mul 65535 1024 50
one "silu_mul d14336" $E $O/silu_mul_bf16_d14336.co silu_mul_bf16_d14336 silu_mul 12483 14336 50
one "silu_mul d24 corr" $E $O/silu_mul_bf16_d24.co silu_mul_bf16_d24 silu_mul 513 24 5
one "gelu_tanh d4096" $E $O/gelu_tanh_mul_bf16_d4096.co gelu_tanh_mul_bf16_d4096 gelu_tanh_mul 43690 4096 50
one "add_bf16" $E $O/add_bf16.co add_bf16 add 87381 2048 50
one "cvt_f32_bf16" $C $O/cvt_f32_bf16.co cvt_f32_bf16 f32_bf16 268435456 50
one "cvt_bf16_f32" $C $O/cvt_bf16_f32.co cvt_bf16_f32 bf16_f32 268435456 50
one "quant_static" $C $O/quant_bf16_fp8_static.co quant_bf16_fp8_static quant_static 536870912 50
one "rmsnorm n4096" $RW $O/rmsnorm_bf16_n4096.co rmsnorm_bf16_n4096 rmsnorm 65536 4096 50
one "rmsnorm n1000" $RW $O/rmsnorm_bf16_n1000.co rmsnorm_bf16_n1000 rmsnorm 262144 1000 50
one "rmsnorm n200" $RW $O/rmsnorm_bf16_n200.co rmsnorm_bf16_n200 rmsnorm 1000000 200 50
one "add_rmsnorm n4096" $RW $O/add_rmsnorm_bf16_n4096.co add_rmsnorm_bf16_n4096 add_rmsnorm 32768 4096 50
one "layernorm n4096" $RW $O/layernorm_bf16_n4096.co layernorm_bf16_n4096 layernorm 65536 4096 50
one "layernorm n776" $RW $O/layernorm_bf16_n776.co layernorm_bf16_n776 layernorm 300000 776 50
one "softmax n4096" $RW $O/softmax_bf16_n4096.co softmax_bf16_n4096 softmax 65536 4096 50
one "softmax n1000" $RW $O/softmax_bf16_n1000.co softmax_bf16_n1000 softmax 262144 1000 50
one "softmax n8192" $RW $O/softmax_bf16_n8192.co softmax_bf16_n8192 softmax 32768 8192 50
one "quant_token n4096" $Q $O/quant_token_bf16_fp8_n4096.co quant_token_bf16_fp8_n4096 token 131072 4096 50
one "quant_token n1000" $Q $O/quant_token_bf16_fp8_n1000.co quant_token_bf16_fp8_n1000 token 524288 1000 50
one "quant_group128 n4096" $Q $O/quant_group128_bf16_fp8_n4096.co quant_group128_bf16_fp8_n4096 group128 131072 4096 50
one "quant_group128 n640 corr" $Q $O/quant_group128_bf16_fp8_n640.co quant_group128_bf16_fp8_n640 group128 19 640 5
one "quant_token_int8 n4096" $Q $O/quant_token_bf16_int8_n4096.co quant_token_bf16_int8_n4096 token_int8 131072 4096 50
one "quant_token_int8 n1000" $Q $O/quant_token_bf16_int8_n1000.co quant_token_bf16_int8_n1000 token_int8 524288 1000 50
one "dequant static" $DQ $O/dequant_fp8_bf16_static.co dequant_fp8_bf16_static static 524288 1024 50
one "dequant token n4096" $DQ $O/dequant_fp8_bf16_token_n4096.co dequant_fp8_bf16_token_n4096 token 65535 4096 50
one "dequant group128 n4096" $DQ $O/dequant_fp8_bf16_group128_n4096.co dequant_fp8_bf16_group128_n4096 group128 65535 4096 50
one "dequant token n1000 (odd tail)" $DQ $O/dequant_fp8_bf16_token_n1000.co dequant_fp8_bf16_token_n1000 token 65535 1000 50
one "silu_mul_quant d4096" $Q $O/silu_mul_quant_fp8_d4096.co silu_mul_quant_fp8_d4096 silu_token 65536 4096 50
one "silu_mul_quant d1000" $Q $O/silu_mul_quant_fp8_d1000.co silu_mul_quant_fp8_d1000 silu_token 262144 1000 50
one "rope_neox h32 d128" $RP $O/rope_neox_bf16_h32_d128.co rope_neox_bf16_h32_d128 neox 65536 32 128 50
one "rope_gptj h32 d128" $RP $O/rope_gptj_bf16_h32_d128.co rope_gptj_bf16_h32_d128 gptj 65536 32 128 50
one "rope_neox h8 d128" $RP $O/rope_neox_bf16_h8_d128.co rope_neox_bf16_h8_d128 neox 262144 8 128 50
one "rope_neox h40 d64 corr" $RP $O/rope_neox_bf16_h40_d64.co rope_neox_bf16_h40_d64 neox 17 40 64 5
one "cache h8 d128" $CA $O/reshape_cache_flash_bf16_h8_d128.co reshape_cache_flash_bf16_h8_d128 131072 8 128 150000 50
one "cache h5 d64 corr" $CA $O/reshape_cache_flash_bf16_h5_d64.co reshape_cache_flash_bf16_h5_d64 77 5 64 200 5
one "topk e128 k8 T=65536" $TK $O/topk_softmax_bf16_e128_k8.co topk_softmax_bf16_e128_k8 65536 128 8 plain 100
one "topk renorm e256 k8 T=65536" $TK $O/topk_softmax_renorm_bf16_e256_k8.co topk_softmax_renorm_bf16_e256_k8 65536 256 8 renorm 100
true