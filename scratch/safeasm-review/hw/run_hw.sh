#!/usr/bin/env bash
# Raw-asm hardware probes for the ISA rules encoded by amdgpu-safeasm (see the review section of docs/safe-asm-design.md).
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
    v_mov_b32_e32 v48, 0
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
ZERO_D=''
for i in $(seq 32 47); do ZERO_D="$ZERO_D    v_mov_b32_e32 v$i, 0
"; done
F16_ON='    v_mov_b32_e32 v8, 0x3C003C00
    v_mov_b32_e32 v9, 0x3C003C00
    v_mov_b32_e32 v10, 0x3C003C00
    v_mov_b32_e32 v11, 0x3C003C00
    v_mov_b32_e32 v12, 0x3C003C00
    v_mov_b32_e32 v13, 0x3C003C00
    v_mov_b32_e32 v14, 0x3C003C00
    v_mov_b32_e32 v15, 0x3C003C00
    v_mov_b32_e32 v16, 0x3C003C00
    v_mov_b32_e32 v17, 0x3C003C00'
F16_ZERO_A='    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0'
FP8='    v_mov_b32_e32 v8, 0x38383838
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
    s_nop 7
    s_nop 3'
WAIT20='    s_nop 7
    s_nop 7
    s_nop 3'
mk() {
  { sed "s/__NAME__/$1/g" <<<"$HDR"; printf '%s\n' "$2"; sed "s/__NAME__/$1/g" <<<"$FTR"; } > "out/$1.s"
  if ! "$ASM_ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "out/$1.co" "out/$1.s" 2> "out/$1.err"; then
    echo "ASSEMBLE FAIL $1"; head -6 "out/$1.err"; return 1
  fi
}
# 32x32x64 f8f6f4 scaled MFMA (A = 8 dwords v[8:15], B = 8 dwords v[16:23], D = 16 dwords v[32:47], scales v24/v25)
# 1) VALU -> MFMA operand wait: A overwritten with ones immediately before the MFMA (stale read -> 0)
for PAD in 1 7; do
  mk "valu2mfma_pad$PAD" "$FP8
$ZERO_D
$WAIT20
    v_mov_b32_e32 v8, 0x38383838
    v_mov_b32_e32 v9, 0x38383838
    v_mov_b32_e32 v10, 0x38383838
    v_mov_b32_e32 v11, 0x38383838
    v_mov_b32_e32 v12, 0x38383838
    v_mov_b32_e32 v13, 0x38383838
    v_mov_b32_e32 v14, 0x38383838
    v_mov_b32_e32 v15, 0x38383838
    s_nop $PAD
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
$WAIT20
    global_store_dword v[4:5], v32, off"
done
# 2) WAR: overwrite an A dword immediately after the MFMA issue vs after 20 states
mk "war_nopad" "$FP8
$ZERO_D
$WAIT20
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
    v_mov_b32_e32 v8, 0
$WAIT20
    global_store_dword v[4:5], v32, off"
mk "war_pad20" "$FP8
$ZERO_D
$WAIT20
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
$WAIT20
    v_mov_b32_e32 v8, 0
$WAIT20
    global_store_dword v[4:5], v32, off"
# 4) RAW: read MFMA D via VALU immediately vs after 20 states
mk "rawread_nopad" "$FP8
$ZERO_D
$WAIT20
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
    v_add_f32_e32 v50, v32, v49
$WAIT20
    global_store_dword v[4:5], v50, off"
mk "rawread_pad20" "$FP8
$ZERO_D
$WAIT20
    v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[8:15], v[16:23], 0, v24, v25
$WAIT20
    v_add_f32_e32 v50, v32, v49
$WAIT20
    global_store_dword v[4:5], v50, off"
echo "[hw] assembling done; building host"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ hw_host.cpp -o out/hw_host -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
for co in out/*.co; do n="$(basename "$co" .co)"; ./out/hw_host "$co" "$n"; done
