declare void @llvm.amdgcn.s.barrier()
define amdgpu_kernel void @k(ptr addrspace(1) %p) {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %l = getelementptr inbounds [512 x i32], ptr addrspace(3) @lds, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %l
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %v = load i32, ptr addrspace(3) %l
  %q = getelementptr i32, ptr addrspace(1) %p, i32 %t
  store i32 %v, ptr addrspace(1) %q
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x()
@lds = internal addrspace(3) global [512 x i32] poison