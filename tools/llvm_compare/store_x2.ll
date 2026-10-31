define amdgpu_kernel void @k(ptr addrspace(1) %p, <2 x i32> %v) {
  store <2 x i32> %v, ptr addrspace(1) %p
  ret void
}