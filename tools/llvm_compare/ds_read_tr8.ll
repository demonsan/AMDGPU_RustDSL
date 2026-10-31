declare <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3))
define amdgpu_kernel void @k(ptr addrspace(1) %p, ptr addrspace(3) %l) {
  %r = call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %l)
  store <2 x i32> %r, ptr addrspace(1) %p
  ret void
}