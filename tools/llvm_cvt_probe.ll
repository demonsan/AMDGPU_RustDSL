; Ground truth for the gfx950 conversion instructions (docs/op-coverage-tileops.md, A2): llc prints the assembler syntax and the MIR
; operand lists our emit / LLVM-tail rows must match.  bash: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 tools/llvm_cvt_probe.ll -o -
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @cvts(ptr addrspace(1) %out, i32 %x, float %f, float %g, float %s, i32 %old) {
  %p = getelementptr i32, ptr addrspace(1) %out, i32 0
  %a0 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %x, i1 false)
  %a1 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %x, i1 true)
  %a2 = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %x, i1 true)
  %b0 = call float @llvm.amdgcn.cvt.f32.fp8(i32 %x, i32 2)
  %b1 = call float @llvm.amdgcn.cvt.f32.bf8(i32 %x, i32 3)
  %c0 = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %f, float %g, i32 %old, i1 false)
  %c1 = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %f, float %g, i32 %c0, i1 true)
  %d0 = call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp8(i32 %x, float %s, i1 false)
  %d1 = call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp8(i32 %x, float %s, i1 true)
  %d2 = call <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp8(i32 %x, float %s, i1 true)
  %d3 = call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.bf8(i32 %x, float %s, i1 true)
  %e0 = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32 %x, float %s, i1 true)
  %e1 = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32 %x, float %s, i1 false)
  %f0 = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp4(i32 %x, float %s, i32 2)
  %f1 = call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(i32 %x, float %s, i32 3)
  %f2 = call <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp4(i32 %x, float %s, i32 1)
  %g0 = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %f, float %g, float %s, i32 2)
  %h0 = call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(<2 x i16> zeroinitializer, float %f, float %g, float %s, i1 true)
  %h1 = call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(<2 x i16> zeroinitializer, float %f, float %g, float %s, i1 false)
  %i0 = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %f, i32 2, i32 %old)
  %j0 = call <2 x i16> @llvm.amdgcn.cvt.pk.i16(i32 %x, i32 %old)
  %j1 = call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %x, i32 %old)
  store volatile <2 x float> %a0, ptr addrspace(1) %p
  store volatile <2 x float> %a1, ptr addrspace(1) %p
  store volatile <2 x float> %a2, ptr addrspace(1) %p
  store volatile float %b0, ptr addrspace(1) %p
  store volatile float %b1, ptr addrspace(1) %p
  store volatile i32 %c1, ptr addrspace(1) %p
  store volatile <2 x bfloat> %d0, ptr addrspace(1) %p
  store volatile <2 x bfloat> %d1, ptr addrspace(1) %p
  store volatile <2 x half> %d2, ptr addrspace(1) %p
  store volatile <2 x bfloat> %d3, ptr addrspace(1) %p
  store volatile <2 x float> %e0, ptr addrspace(1) %p
  store volatile <2 x float> %e1, ptr addrspace(1) %p
  store volatile <2 x float> %f0, ptr addrspace(1) %p
  store volatile <2 x bfloat> %f1, ptr addrspace(1) %p
  store volatile <2 x half> %f2, ptr addrspace(1) %p
  store volatile i32 %g0, ptr addrspace(1) %p
  store volatile <2 x i16> %h0, ptr addrspace(1) %p
  store volatile <2 x i16> %h1, ptr addrspace(1) %p
  store volatile i32 %i0, ptr addrspace(1) %p
  store volatile <2 x i16> %j0, ptr addrspace(1) %p
  store volatile <2 x i16> %j1, ptr addrspace(1) %p
  ret void
}

declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1)
declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)
declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32)
declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)
declare <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp8(i32, float, i1)
declare <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp8(i32, float, i1)
declare <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.bf8(i32, float, i1)
declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32, float, i1)
declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32, float, i1)
declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp4(i32, float, i32)
declare <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(i32, float, i32)
declare <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp4(i32, float, i32)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32)
declare <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(<2 x i16>, float, float, float, i1)
declare <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(<2 x i16>, float, float, float, i1)
declare i32 @llvm.amdgcn.cvt.pk.u8.f32(float, i32, i32)
declare <2 x i16> @llvm.amdgcn.cvt.pk.i16(i32, i32)
declare <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32, i32)
