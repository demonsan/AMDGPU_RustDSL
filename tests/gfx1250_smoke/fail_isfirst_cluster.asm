# EXPECT-FAIL: s_barrier_signal_isfirst with the cluster user barrier id (-3) is rejected by the assembler
# (upstream llvm/lib/Target/AMDGPU/AsmParser/AMDGPUAsmParser.cpp:5585 @274ab686)
.text
s_barrier_signal_isfirst -3
