declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a, float %b) {
  %r = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %a, float %b, i32 0, i1 0)
  store i32 %r, ptr addrspace(1) %p
  ret void
}