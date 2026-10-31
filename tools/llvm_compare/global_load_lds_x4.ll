declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, i32 immarg, i32 immarg, i32 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %g, ptr addrspace(3) %l) {
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %g, ptr addrspace(3) %l, i32 16, i32 0, i32 0)
  ret void
}