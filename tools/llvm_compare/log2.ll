declare float @llvm.amdgcn.log.f32(float)
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a) {
  %r = call float @llvm.amdgcn.log.f32(float %a)
  store float %r, ptr addrspace(1) %p
  ret void
}