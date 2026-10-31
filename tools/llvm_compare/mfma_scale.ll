declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4(<8 x i32>, <8 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)
define amdgpu_kernel void @k(ptr addrspace(1) %p, <8 x i32> %a, <8 x i32> %b, i32 %sa, i32 %sb) {
  %r = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4(<8 x i32> %a, <8 x i32> %b, <16 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 %sa, i32 0, i32 %sb)
  store <16 x float> %r, ptr addrspace(1) %p
  ret void
}