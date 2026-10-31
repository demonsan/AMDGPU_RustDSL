define amdgpu_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %c) {
  %old = atomicrmw add ptr addrspace(1) %c, i32 1 monotonic
  store i32 %old, ptr addrspace(1) %p
  ret void
}