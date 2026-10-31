declare <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32>, i32, i32, i32 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %p, <4 x i32> %rs, i32 %vo, i32 %so) {
  %r = call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %rs, i32 %vo, i32 %so, i32 0)
  store <4 x i32> %r, ptr addrspace(1) %p
  ret void
}