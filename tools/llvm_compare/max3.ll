declare float @llvm.maxnum.f32(float, float)
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a, float %b, float %c) {
  %t = call float @llvm.maxnum.f32(float %a, float %b)
  %r = call float @llvm.maxnum.f32(float %t, float %c)
  store float %r, ptr addrspace(1) %p
  ret void
}