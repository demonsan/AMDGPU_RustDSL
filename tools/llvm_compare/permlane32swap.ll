declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg)
define amdgpu_kernel void @k(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %r = call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %a, i32 %b, i1 0, i1 0)
  %x = extractvalue { i32, i32 } %r, 0
  %y = extractvalue { i32, i32 } %r, 1
  %s = add i32 %x, %y
  store i32 %s, ptr addrspace(1) %p
  ret void
}