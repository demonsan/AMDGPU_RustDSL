#!/usr/bin/env bash
# Batch-2 matrix: emit + assemble every kernel, then correctness (small odd shapes) and bandwidth (~1 GB working sets) with the host drivers in tests/gpu. One line per case.
# usage (on a gfx950 node, from the repo root, with $AMDC = a built amdc): bash tools/b2_matrix.sh [out]   (the vendor harness is separate: tools/vendor_membound.sh)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4}
A=${AMDC:-/tmp/target-b2/debug/amdc}; R=${ROCM:-/shared/apps/ubuntu/opt/rocm-7.0.2}; CL=${CLANG:-/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang}; O=${1:-build/b2}
mkdir -p $O
for h in eltwise rows quant cast rope cache topk dequant cache_blocked moe_sorting sampling moe_reduce gated cache_fp8 conv1d_update moe_gate topk_per_row conv1d_fwd cache_blocked_fp8 qknorm_rope_kv mxnorm mla_cache eltwise_bias mxfp6; do [ -x $O/${h}_host ] || $R/bin/amdclang++ -x hip -O2 --offload-arch=gfx950 --rocm-path=$R -std=c++17 tests/gpu/${h}_host.cpp -o $O/${h}_host -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib 2>&1 | grep error; done
mk() { [ -f $O/$1.co ] && [ $O/$1.co -nt $A ] && return 0; $A emit $1 --backend=own --out $O/$1.s > $O/$1.log 2>&1 || { echo "EMIT FAIL $1"; head -4 $O/$1.log; return 1; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o $O/$1.co $O/$1.s; }
one() { lab=$1; shift; out=$("$@" 2>&1); echo "$lab | $(echo "$out" | grep -E '^check' | sed -E 's/.*(bad[a-z_]*=[0-9]+).*(OK|FAIL)$/\1 \2/' | head -1) | $(echo "$out" | grep -E '^timing' | sed -E 's/.*ms\/iter[^,]*, ([0-9.]+) GB\/s.*/\1 GB\/s/; s/^timing: ([0-9.]+) ms\/iter \(([0-9.]+) us\).*/\2 us/' | head -1)"; }
CB=$O/cache_blocked_host; DQ=$O/dequant_host; E=$O/eltwise_host; RW=$O/rows_host; Q=$O/quant_host; C=$O/cast_host; RP=$O/rope_host; CA=$O/cache_host; TK=$O/topk_host
for k in silu_mul_bf16_d4096 silu_mul_bf16_d1024 silu_mul_bf16_d14336 silu_mul_bf16_d24 gelu_tanh_mul_bf16_d4096 silu_mul_limit_bf16_d4096 swiglu_oai_mul_bf16_d4096 silu_bf16 gelu_tanh_bf16 relu2_bf16 sigmoid_bf16 add_bf16 cvt_f32_bf16 cvt_bf16_f32 quant_bf16_fp8_static rmsnorm_bf16_n4096 rmsnorm_bf16_n1000 rmsnorm_bf16_n200 add_rmsnorm_bf16_n4096 layernorm_bf16_n4096 layernorm_bf16_n776 softmax_bf16_n4096 softmax_bf16_n1000 softmax_bf16_n8192 quant_token_bf16_fp8_n4096 quant_token_bf16_fp8_n14336 quant_token_bf16_fp8_n32768 quant_token_bf16_int8_n14336 silu_mul_quant_fp8_d14336 situv2_mul_quant_fp8_d14336 silu_mul_quant_fp8_d32768 quant_token_bf16_fp8_n1000 quant_group128_bf16_fp8_n4096 quant_group128_bf16_fp8_n640 quant_token_bf16_int8_n4096 quant_token_bf16_int8_n1000 dequant_fp8_bf16_static dequant_fp8_bf16_token_n4096 dequant_fp8_bf16_group128_n4096 dequant_fp8_bf16_token_n1000 silu_mul_quant_fp8_d4096 situv2_mul_quant_fp8_d4096 silu_mul_quant_fp8_d1000 rope_neox_bf16_h32_d128 rope_gptj_bf16_h32_d128 rope_neox_bf16_h8_d128 rope_neox_bf16_h40_d64 reshape_cache_flash_bf16_h8_d128 reshape_cache_flash_bf16_h5_d64 reshape_cache_flash_fp8_h8_d128 reshape_cache_blocked_fp8_h8_d128_b16_i64 qknorm_rope_kvcache_neox_bf16_h8_d128 qknorm_rope_kvcache_neox_fp8_h8_d128_i64 rmsnorm_quant_mxfp4_bf16_k4096 quant_mxfp6_e2m3_bf16_k4096 quant_mxfp6_e3m2_bf16_k4096 concat_cache_mla_bf16_r512_p64 concat_cache_mla_seg_fp8_r512_p64_b64_i64 silu_mul_bias_bf16_d4096 gelu_mul_bias_bf16_d2048 gelu_bf16 gelu_mul_bf16_d4096 reshape_cache_flash_bf16_h8_d128_i64 gated_rmsnorm_bf16_n128 gated_rmsnorm_fp8_n128 gated_rmsnorm_bf16_n4096 gated_rmsnorm_fp8_n4096 add_rmsnorm_fp8_n4096 add_rmsnorm_fp8_n1000 qknorm_rope_neox_bf16_h40_d128 causal_conv1d_update_bf16_w4_s3_idx_silu moe_fused_gate_bf16_e256_g8_tg4_k8 causal_conv1d_fwd_bf16_w4_silu causal_conv1d_fwd_bf16_w4_silu_sched causal_conv1d_fwd_split_bf16_w4_silu topk_per_row_decode_f32_k2048 topk_per_row_prefill_f32_k128 reshape_cache_blocked_bf16_h8_d128_b16 reshape_cache_blocked_bf16_h8_d64_b64 topk_softmax_bf16_e128_k8 topk_softmax_renorm_bf16_e256_k8 moe_sorting_e128_k8_b32 moe_sorting_e256_k8_b128 moe_sorting_e385_k6_b32 moe_sorting_e64_k2_b16 moe_reduce_bf16_k8_d7168 moe_reduce_bf16_k8_d4096 moe_combine_bf16_k8_d4096 argmax_bf16_v32000 argmax_bf16_v128256 sample_bf16_v32000 sample_bf16_v128256 sample_bf16_v1000; do mk $k; done
one "silu_mul d4096 corr+perf" $E $O/silu_mul_bf16_d4096.co silu_mul_bf16_d4096 silu_mul 43690 4096 50
one "silu_mul d1024" $E $O/silu_mul_bf16_d1024.co silu_mul_bf16_d1024 silu_mul 65535 1024 50
one "silu_mul d14336" $E $O/silu_mul_bf16_d14336.co silu_mul_bf16_d14336 silu_mul 12483 14336 50
one "silu_mul d24 corr" $E $O/silu_mul_bf16_d24.co silu_mul_bf16_d24 silu_mul 513 24 5
one "gelu_tanh d4096" $E $O/gelu_tanh_mul_bf16_d4096.co gelu_tanh_mul_bf16_d4096 gelu_tanh_mul 43690 4096 50
one "silu_mul_limit d4096 (limit 7)" $E $O/silu_mul_limit_bf16_d4096.co silu_mul_limit_bf16_d4096 silu_mul_limit 43690 4096 50 7.0
one "swiglu_oai_mul d4096" $E $O/swiglu_oai_mul_bf16_d4096.co swiglu_oai_mul_bf16_d4096 swiglu_oai 43690 4096 50
one "silu (unary)" $E $O/silu_bf16.co silu_bf16 silu 87381 2048 50
one "gelu_tanh (unary)" $E $O/gelu_tanh_bf16.co gelu_tanh_bf16 gelu_tanh 87381 2048 50
one "relu2 (unary)" $E $O/relu2_bf16.co relu2_bf16 relu2 87381 2048 50
one "sigmoid (unary)" $E $O/sigmoid_bf16.co sigmoid_bf16 sigmoid 87381 2048 50
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
one "quant_token n14336 (long row)" $Q $O/quant_token_bf16_fp8_n14336.co quant_token_bf16_fp8_n14336 token 36000 14336 50
one "quant_token n14336 corr" $Q $O/quant_token_bf16_fp8_n14336.co quant_token_bf16_fp8_n14336 token 37 14336 5
one "quant_token n32768 (long row)" $Q $O/quant_token_bf16_fp8_n32768.co quant_token_bf16_fp8_n32768 token 16384 32768 50
one "quant_token_int8 n14336 (long row)" $Q $O/quant_token_bf16_int8_n14336.co quant_token_bf16_int8_n14336 token_int8 36000 14336 50
one "silu_mul_quant d14336 (long row)" $Q $O/silu_mul_quant_fp8_d14336.co silu_mul_quant_fp8_d14336 silu_token 17000 14336 50
one "silu_mul_quant d14336 corr" $Q $O/silu_mul_quant_fp8_d14336.co silu_mul_quant_fp8_d14336 silu_token 37 14336 5
one "silu_mul_quant d32768 (long row)" $Q $O/silu_mul_quant_fp8_d32768.co silu_mul_quant_fp8_d32768 silu_token 7500 32768 50
one "situv2_mul_quant d14336 (long row)" $Q $O/situv2_mul_quant_fp8_d14336.co situv2_mul_quant_fp8_d14336 situ_token 17000 14336 30
one "quant_token n1000" $Q $O/quant_token_bf16_fp8_n1000.co quant_token_bf16_fp8_n1000 token 524288 1000 50
one "situv2_mul_quant d4096" $Q $O/situv2_mul_quant_fp8_d4096.co situv2_mul_quant_fp8_d4096 situ_token 65536 4096 30
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
one "cache_blocked h8 d128 b16 random slots" $CB $O/reshape_cache_blocked_bf16_h8_d128_b16.co reshape_cache_blocked_bf16_h8_d128_b16 131072 8 128 16 9400 50
CONTIG=1 one "cache_blocked h8 d128 b16 contiguous slots" $CB $O/reshape_cache_blocked_bf16_h8_d128_b16.co reshape_cache_blocked_bf16_h8_d128_b16 131072 8 128 16 9400 50
one "cache_blocked h8 d64 b64 corr" $CB $O/reshape_cache_blocked_bf16_h8_d64_b64.co reshape_cache_blocked_bf16_h8_d64_b64 131 8 64 64 6 5
MS=$O/moe_sorting_host
one "moe_sorting e128 k8 b32 M=37" $MS $O/moe_sorting_e128_k8_b32.co moe_sorting_e128_k8_b32 37 128 8 32 20
one "moe_sorting e128 k8 b32 M=1001 skew+invalid" $MS $O/moe_sorting_e128_k8_b32.co moe_sorting_e128_k8_b32 1001 128 8 32 20 1
one "moe_sorting e64 k2 b16 M=513" $MS $O/moe_sorting_e64_k2_b16.co moe_sorting_e64_k2_b16 513 64 2 16 20
one "moe_sorting e256 k8 b128 M=4096" $MS $O/moe_sorting_e256_k8_b128.co moe_sorting_e256_k8_b128 4096 256 8 128 20
one "moe_sorting e385 k6 b32 M=777" $MS $O/moe_sorting_e385_k6_b32.co moe_sorting_e385_k6_b32 777 385 6 32 20
SP=$O/sampling_host
one "argmax v32000 B=16384" $SP $O/argmax_bf16_v32000.co argmax_bf16_v32000 argmax 16384 32000 30
one "argmax v128256 B=4096" $SP $O/argmax_bf16_v128256.co argmax_bf16_v128256 argmax 4096 128256 30
one "sample v32000 B=16384" $SP $O/sample_bf16_v32000.co sample_bf16_v32000 sample 16384 32000 30
one "sample v128256 B=4096" $SP $O/sample_bf16_v128256.co sample_bf16_v128256 sample 4096 128256 30
one "sample v1000 corr" $SP $O/sample_bf16_v1000.co sample_bf16_v1000 sample 41 1000 5
MRH=$O/moe_reduce_host
one "moe_reduce k8 d7168 T=16384" $MRH $O/moe_reduce_bf16_k8_d7168.co moe_reduce_bf16_k8_d7168 reduce 16384 8 7168 20
one "moe_reduce k8 d4096 T=32768" $MRH $O/moe_reduce_bf16_k8_d4096.co moe_reduce_bf16_k8_d4096 reduce 32768 8 4096 20
one "moe_combine k8 d4096 T=32768" $MRH $O/moe_combine_bf16_k8_d4096.co moe_combine_bf16_k8_d4096 sum 32768 8 4096 20
one "moe_reduce k6 d2048 corr" $MRH $O/moe_reduce_bf16_k8_d7168.co moe_reduce_bf16_k8_d7168 reduce 37 8 7168 5
GH=$O/gated_host; FH=$O/cache_fp8_host
one "gated_rmsnorm bf16 n128 (heads)" $GH $O/gated_rmsnorm_bf16_n128.co gated_rmsnorm_bf16_n128 bf16 4000000 128 20
one "gated_rmsnorm fp8 n128 (heads)" $GH $O/gated_rmsnorm_fp8_n128.co gated_rmsnorm_fp8_n128 fp8 4000000 128 20
one "gated_rmsnorm bf16 n4096" $GH $O/gated_rmsnorm_bf16_n4096.co gated_rmsnorm_bf16_n4096 bf16 65536 4096 20
one "gated_rmsnorm fp8 n4096" $GH $O/gated_rmsnorm_fp8_n4096.co gated_rmsnorm_fp8_n4096 fp8 65536 4096 20
one "cache_fp8 h8 d128" $FH $O/reshape_cache_flash_fp8_h8_d128.co reshape_cache_flash_fp8_h8_d128 131072 8 128 150000 30
CVH=$O/conv1d_update_host
one "add_rmsnorm_fp8 n4096" $GH $O/add_rmsnorm_fp8_n4096.co add_rmsnorm_fp8_n4096 addfp8 65536 4096 20
one "add_rmsnorm_fp8 n1000" $GH $O/add_rmsnorm_fp8_n1000.co add_rmsnorm_fp8_n1000 addfp8 262144 1000 20
one "qknorm_rope neox h40 d128" $RP $O/qknorm_rope_neox_bf16_h40_d128.co qknorm_rope_neox_bf16_h40_d128 neox 65536 40 128 20
one "conv1d_update w4 s3 idx silu" $CVH $O/causal_conv1d_update_bf16_w4_s3_idx_silu.co causal_conv1d_update_bf16_w4_s3_idx_silu 2048 8192 4 3 1 1 30
MGH=$O/moe_gate_host
one "moe_fused_gate e256 g8 tg4 k8 M=131072" $MGH $O/moe_fused_gate_bf16_e256_g8_tg4_k8.co moe_fused_gate_bf16_e256_g8_tg4_k8 131072 256 8 4 8 30
one "moe_fused_gate e256 g8 tg4 k8 M=1001 (odd)" $MGH $O/moe_fused_gate_bf16_e256_g8_tg4_k8.co moe_fused_gate_bf16_e256_g8_tg4_k8 1001 256 8 4 8 5
TR=$O/topk_per_row_host
one "topk_per_row decode k2048 B=256 L<=64K" $TR $O/topk_per_row_decode_f32_k2048.co topk_per_row_decode_f32_k2048 decode 256 1 65536 65536 2048 10
one "topk_per_row decode k2048 odd (ragged, ties)" $TR $O/topk_per_row_decode_f32_k2048.co topk_per_row_decode_f32_k2048 decode 13 2 8192 6000 2048 3 1
one "topk_per_row prefill k128 (ties)" $TR $O/topk_per_row_prefill_f32_k128.co topk_per_row_prefill_f32_k128 prefill 33 1 4096 3000 128 3 1
CFH=$O/conv1d_fwd_host
one "conv1d_fwd w4 silu B=128 L=512 uniform" $CFH $O/causal_conv1d_fwd_bf16_w4_silu.co causal_conv1d_fwd_bf16_w4_silu plain 128 8192 4 0 1 512 10 1
one "conv1d_fwd split w4 silu B=128 L=512 uniform" $CFH $O/causal_conv1d_fwd_split_bf16_w4_silu.co causal_conv1d_fwd_split_bf16_w4_silu split 128 8192 4 2048 1 512 10 1
one "conv1d_fwd w4 silu ragged B=256 (grid)" $CFH $O/causal_conv1d_fwd_bf16_w4_silu.co causal_conv1d_fwd_bf16_w4_silu plain 256 8192 4 0 1 1024 10
one "conv1d_fwd w4 silu ragged B=256 (_sched)" $CFH $O/causal_conv1d_fwd_bf16_w4_silu_sched.co causal_conv1d_fwd_bf16_w4_silu_sched plain 256 8192 4 0 1 1024 10
one "conv1d_fwd split w4 silu ragged + states" $CFH $O/causal_conv1d_fwd_split_bf16_w4_silu.co causal_conv1d_fwd_split_bf16_w4_silu split 19 1536 4 512 1 90 3
CBF=$O/cache_blocked_fp8_host
one "cache_blocked fp8 per-token h8 d128 b16 (i64 slots)" $CBF $O/reshape_cache_blocked_fp8_h8_d128_b16_i64.co reshape_cache_blocked_fp8_h8_d128_b16_i64 8192 8 128 16 1024 1 20
QKV=$O/qknorm_rope_kv_host
one "qknorm_rope_kvcache neox bf16 h8 d128" $QKV $O/qknorm_rope_kvcache_neox_bf16_h8_d128.co qknorm_rope_kvcache_neox_bf16_h8_d128 neox 65536 8 128 80000 0 0 20
one "qknorm_rope_kvcache neox fp8 h8 d128 (i64)" $QKV $O/qknorm_rope_kvcache_neox_fp8_h8_d128_i64.co qknorm_rope_kvcache_neox_fp8_h8_d128_i64 neox 65536 8 128 80000 1 1 20
MXN=$O/mxnorm_host
one "rmsnorm_quant_mxfp4 k4096 M=32768" $MXN $O/rmsnorm_quant_mxfp4_bf16_k4096.co rmsnorm_quant_mxfp4_bf16_k4096 32768 4096 20
MLA=$O/mla_cache_host; EB=$O/eltwise_bias_host
one "mla cache flat bf16 r512 p64" $MLA $O/concat_cache_mla_bf16_r512_p64.co concat_cache_mla_bf16_r512_p64 flat bf16 65536 512 64 16 5000 0 20
one "mla cache seg fp8 r512 p64 b64 (i64)" $MLA $O/concat_cache_mla_seg_fp8_r512_p64_b64_i64.co concat_cache_mla_seg_fp8_r512_p64_b64_i64 seg fp8 65536 512 64 64 1500 1 20
one "silu_mul_bias d4096 E=64" $EB $O/silu_mul_bias_bf16_d4096.co silu_mul_bias_bf16_d4096 silu 32768 4096 64 20
one "gelu_mul_bias d2048 E=64" $EB $O/gelu_mul_bias_bf16_d2048.co gelu_mul_bias_bf16_d2048 gelu 32768 2048 64 20
one "gelu (erf) unary" $E $O/gelu_bf16.co gelu_bf16 gelu 87381 2048 50
one "gelu_mul (erf) d4096" $E $O/gelu_mul_bf16_d4096.co gelu_mul_bf16_d4096 gelu_mul 43690 4096 50
MX6=$O/mxfp6_host
one "mxfp6 e2m3 k4096 M=32768" $MX6 $O/quant_mxfp6_e2m3_bf16_k4096.co quant_mxfp6_e2m3_bf16_k4096 e2m3 32768 4096 20
one "mxfp6 e3m2 k4096 M=32768" $MX6 $O/quant_mxfp6_e3m2_bf16_k4096.co quant_mxfp6_e3m2_bf16_k4096 e3m2 32768 4096 20
one "topk e128 k8 T=65536" $TK $O/topk_softmax_bf16_e128_k8.co topk_softmax_bf16_e128_k8 65536 128 8 plain 100
one "topk renorm e256 k8 T=65536" $TK $O/topk_softmax_renorm_bf16_e256_k8.co topk_softmax_renorm_bf16_e256_k8 65536 256 8 renorm 100
true