target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k(ptr addrspace(1) %out, i32 %old, float %f, float %g, float %s) {
  %a = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %old, float %f, float %g, float %s, i32 1)
  %b = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %a, float %f, float %g, float %s, i32 3)
  %c = call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp4(i32 %b, float %s, i32 1)
  store volatile i32 %b, ptr addrspace(1) %out
  store volatile <2 x float> %c, ptr addrspace(1) %out
  ret void
}
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32)
declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp4(i32, float, i32)
