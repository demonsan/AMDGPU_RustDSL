define amdgpu_kernel void @k(ptr addrspace(1) %p, <2 x float> %a, <2 x float> %b) {
  %r = fadd <2 x float> %a, %b
  store <2 x float> %r, ptr addrspace(1) %p
  ret void
}