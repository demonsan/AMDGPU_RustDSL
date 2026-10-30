# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者和自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部 SSA / MIR，自己做寄存器分配、hazard / wait 插入和汇编发射，输出 `.s`，再由 ROCm 自带的 `clang` **只做汇编**得到 `.co`。LLVM 源码不修改，LLVM 的设备端 CodeGen（寄存器分配、调度）不参与。

## 当前状态（已在 MI355X / gfx950 真机上验证）

| 内容 | 状态 |
|---|---|
| Rust 类型化 DSL、IR 验证器、MIR、线性扫描 RA、hazard / wait 插入、asm 发射 | 已实现，`cargo test` 11 项 |
| 端到端：DSL → `.s` → `.co` → HIP 启动 → 数值比对 | 已打通，见 `tests/gpu/` |
| `attention_fwd`（fp16，示例） | 对 CPU 参考最大绝对误差 ≈ 6e-5 |
| MXFP8 FMHA：`fmha_fwd_hd128_mxfp8_t` | 与厂商 `fwd_hd128_mxfp8.co` 使用**相同的 656 字节 kernarg ABI 和数据布局**，15 个形状逐元素对拍通过；B1H8S2048 上 291 TFLOP/s（llvm-tail；厂商 514，差 1.76 倍；快速路径 `fmha_fwd_hd128_mxfp8_vs128` 见下），对 CPU 的平均误差 1.25e-3（优于厂商 2.42e-3） |