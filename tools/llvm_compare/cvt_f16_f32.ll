define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a) {
  %h = fptrunc float %a to half
  %f = fpext half %h to float
  store float %f, ptr addrspace(1) %p
  ret void
}