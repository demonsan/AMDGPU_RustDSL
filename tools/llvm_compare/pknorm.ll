declare <2 x i16> @llvm.amdgcn.cvt.pknorm.u16(float, float)
define amdgpu_kernel void @k(ptr addrspace(1) %p, float %a, float %b) {
  %r = call <2 x i16> @llvm.amdgcn.cvt.pknorm.u16(float %a, float %b)
  store <2 x i16> %r, ptr addrspace(1) %p
  ret void
}