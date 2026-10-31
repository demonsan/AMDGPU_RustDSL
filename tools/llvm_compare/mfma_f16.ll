declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, <4 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %p, <4 x half> %a, <4 x half> %b) {
  %r = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %a, <4 x half> %b, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <16 x float> %r, ptr addrspace(1) %p
  ret void
}