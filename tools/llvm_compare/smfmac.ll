declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32>, <8 x i32>, <4 x float>, i32, i32 immarg, i32 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %p, <4 x i32> %a, <8 x i32> %b, i32 %idx) {
  %r = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32> %a, <8 x i32> %b, <4 x float> zeroinitializer, i32 %idx, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %p
  ret void
}