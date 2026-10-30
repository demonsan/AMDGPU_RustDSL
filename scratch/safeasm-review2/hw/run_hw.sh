#!/usr/bin/env bash
# Raw-asm hardware probes for the wait-state rules the safe-asm verifier encodes: VALU -> MFMA A/B operand, VALU ->
# v_readfirstlane, VALU -> SMFMAC operand, compare -> v_cndmask mask. For every class states = 1,2,3,4 are tried; the
# correct value indicates whether `states` is enough (the verifier requires 2 for all of them).
set -uo pipefail
cd "$(dirname "$0")"
ASM_ROC="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-5}"
mkdir -p out
HDR='.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl __NAME__
.p2align 8
.type __NAME__,@function
__NAME__:
    s_load_dwordx2 s[2:3], s[0:1], 0
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v0, 63, v0
    v_lshlrev_b32_e32 v0, 4, v0
    v_mov_b32_e32 v4, s2
    v_mov_b32_e32 v5, s3
    v_add_co_u32 v4, vcc, v4, v0
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    v_mov_b32_e32 v48, 1
    v_mov_b32_e32 v49, 0'
FTR='    s_endpgm
.size __NAME__, .-__NAME__
.p2align 6
.amdhsa_kernel __NAME__
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: __NAME__
    .symbol: __NAME__.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .args:
      - .name: "o"
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata'
mk() {
  { sed "s/__NAME__/$1/g" <<<"$HDR"; printf '%s\n' "$2"; sed "s/__NAME__/$1/g" <<<"$FTR"; } > "out/$1.s"
  if ! "$ASM_ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "out/$1.co" "out/$1.s" 2> "out/$1.err"; then
    echo "ASSEMBLE FAIL $1"; head -6 "out/$1.err"; return 1
  fi
}
PAD() { if [ "$1" = 1 ]; then echo "    s_nop 0"; else echo "    s_nop $(( $1 - 1 ))"; fi; }
WAIT20='    s_nop 7
    s_nop 7
    s_nop 3'
WAIT8='    s_nop 7'
zero_d() { for i in $(seq 32 47); do echo "    v_mov_b32_e32 v$i, 0"; done; }
# ---- class 1: VALU -> MFMA A/B operand (A stale means zero product) ----
for S in 1 2 3 4; do
  mk "mfma_ab_s$S" "    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v14, 0
    v_mov_b32_e32 v15, 0
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0x38383838
    v_mov_b32_e32 v21, 0x38383838
    v_mov_b32_e32 v22, 0x38383838
    v_mov_b32_e32 v23, 0x38383838
    v_mov_b32_e32 v24, 127
    v_mov_b32_e32 v25, 127
$(zero_d)
$WAIT20
    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
$(PAD $S)
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
$WAIT20
    global_store_dword v[4:5], v32, off"
done
# ---- class 2: VALU -> v_readfirstlane (expected 34 = 0x22 fresh, 17 = 0x11 stale) ----
for S in 1 2 3 4; do
  mk "rfl_s$S" "    v_mov_b32_e32 v8, 0x11
$WAIT8
    s_mov_b32 s8, 0
    v_mov_b32_e32 v8, 0x22
$(PAD $S)
    v_readfirstlane_b32 s8, v8
$WAIT8
    v_mov_b32_e32 v50, s8
$WAIT8
    global_store_dword v[4:5], v50, off"
done
# ---- class 3: VALU -> SMFMAC operand (A stale means zero) ----
for S in 1 2 3 4; do
  mk "smfmac_s$S" "    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0
    v_mov_b32_e32 v32, 0
    v_mov_b32_e32 v33, 0
    v_mov_b32_e32 v34, 0
    v_mov_b32_e32 v35, 0
$WAIT20
    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
$(PAD $S)
    v_smfmac_f32_16x16x128_fp8_fp8 v[32:35], v[8:11], v[12:19], v20
$WAIT20
    global_store_dword v[4:5], v32, off"
done
# ---- class 4: compare -> v_cndmask mask (expected 9 fresh, 7 stale) ----
for S in 1 2 3 4; do
  mk "cndmask_s$S" "    v_mov_b32_e32 v8, 5
    v_mov_b32_e32 v9, 2
    v_mov_b32_e32 v11, 7
    v_mov_b32_e32 v12, 9
    v_cmp_lt_i32_e32 vcc, v8, v9
$WAIT20
    v_mov_b32_e32 v8, 1
    v_cmp_lt_i32_e32 vcc, v8, v9
$(PAD $S)
    v_cndmask_b32_e32 v10, v11, v12, vcc
$WAIT20
    global_store_dword v[4:5], v10, off"
done
PADN() { case "$1" in 20) echo "    s_nop 7
    s_nop 7
    s_nop 3";; 16) echo "    s_nop 7
    s_nop 7
    s_nop 0";; 12) echo "    s_nop 7
    s_nop 3";; 8) echo "    s_nop 7";; 1) echo "    s_nop 0";; *) echo "    s_nop $(( $1 - 1 ))";; esac; }
# ---- class 5: MFMA result read by a VALU op after `states` wait states (how stale is the D really?) ----
for S in 1 8 12 16 20; do
  mk "rawread_s$S" "    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0x38383838
    v_mov_b32_e32 v21, 0x38383838
    v_mov_b32_e32 v22, 0x38383838
    v_mov_b32_e32 v23, 0x38383838
    v_mov_b32_e32 v24, 127
    v_mov_b32_e32 v25, 127
$(zero_d)
$WAIT20
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
$(PADN $S)
    v_add_f32_e32 v50, v32, v49
$WAIT20
    global_store_dword v[4:5], v50, off"
done
# ---- class 6: SMFMAC idx operand written by VALU right before the SMFMAC (stale idx = old sparse pattern) ----
for S in 1 2 3 4; do
  mk "smfmac_idx_s$S" "    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0x44444444
    v_mov_b32_e32 v32, 0
    v_mov_b32_e32 v33, 0
    v_mov_b32_e32 v34, 0
    v_mov_b32_e32 v35, 0
$WAIT20
    v_mov_b32_e32 v20, 0xeeeeeeee
$(PADN $S)
    v_smfmac_f32_16x16x128_fp8_fp8 v[32:35], v[8:11], v[12:19], v20
$WAIT20
    global_store_dword v[4:5], v32, off"
done
# control: idx written long before (no rewrite)
mk "smfmac_idx_ref01" "    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0x44444444
    v_mov_b32_e32 v32, 0
    v_mov_b32_e32 v33, 0
    v_mov_b32_e32 v34, 0
    v_mov_b32_e32 v35, 0
$WAIT20
    v_smfmac_f32_16x16x128_fp8_fp8 v[32:35], v[8:11], v[12:19], v20
$WAIT20
    global_store_dword v[4:5], v32, off"
mk "smfmac_idx_ref23" "    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    v_mov_b32_e32 v16, 0x38383838
    v_mov_b32_e32 v17, 0x38383838
    v_mov_b32_e32 v18, 0x38383838
    v_mov_b32_e32 v19, 0x38383838
    v_mov_b32_e32 v20, 0xeeeeeeee
    v_mov_b32_e32 v32, 0
    v_mov_b32_e32 v33, 0
    v_mov_b32_e32 v34, 0
    v_mov_b32_e32 v35, 0
$WAIT20
    v_smfmac_f32_16x16x128_fp8_fp8 v[32:35], v[8:11], v[12:19], v20
$WAIT20
    global_store_dword v[4:5], v32, off"
echo "[hw2] assembling done; building host"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ hw_host.cpp -o out/hw_host -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
run_all() { for co in out/*.co; do n="$(basename "$co" .co)"; ./out/hw_host "$co" "$n"; done; }
if command -v gpu-lock >/dev/null 2>&1; then
  echo "[hw2] using gpu-lock"
  gpu-lock bash -c "$(declare -f run_all); cd '$PWD'; run_all"
else
  echo "[hw2] gpu-lock not installed on this node; running directly"
  run_all
fi
