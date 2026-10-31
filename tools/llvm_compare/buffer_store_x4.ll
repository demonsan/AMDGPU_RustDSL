declare void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32 immarg)
define amdgpu_kernel void @k(<4 x i32> %rs, <4 x i32> %v, i32 %vo, i32 %so) {
  call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %v, <4 x i32> %rs, i32 %vo, i32 %so, i32 0)
  ret void
}