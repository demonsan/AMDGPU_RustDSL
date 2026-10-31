declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32>, ptr addrspace(3) nocapture, i32 immarg, i32, i32, i32 immarg, i32 immarg)
define amdgpu_kernel void @k(<4 x i32> %rs, ptr addrspace(3) %l, i32 %vo, i32 %so) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rs, ptr addrspace(3) %l, i32 16, i32 %vo, i32 %so, i32 0, i32 0)
  ret void
}