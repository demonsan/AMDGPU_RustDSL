declare float @llvm.amdgcn.exp2.f32(float)
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a) {
  %r = call float @llvm.amdgcn.exp2.f32(float %a)
  store float %r, ptr addrspace(1) %p
  ret void
}