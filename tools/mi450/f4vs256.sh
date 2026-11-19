# ours (TDM stage template, MXFP4) vs the vendor f4gemm on the same idle GPU, alternating (A B A B): same shapes, TFLOPS from each host's own timing loop
set -u
cd "$SRC"
O="$W/f4vs"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
k=${KERNEL:-gemm_g1250_tdm_stage_f4}
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL own"; exit 6; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -Itests/gpu tests/gpu/g1250_f4gemm_host.cpp -o $O/vendor_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL vendor"; exit 6; }
VCO=$BASE/vendor/gfx1250/f4gemm/f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_ps.co
VK=$(grep -a -o '_ZN[0-9A-Za-z_]*E' $VCO | head -1)
echo "GPU: ${IDLE_GPU:-none}  vendor kernel: $VK"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-20}
for shape in ${SHAPES:-"4096 4096 4096" "8192 8192 8192"}; do
  for rep in 1 2; do
    echo "--- shape $shape (rep $rep)"
    echo -n "vendor: "; gpurun $O/vendor_host $VCO $VK $shape 1 50 20000 2>&1 | grep -E '^\{"host"|FAIL|HANG' | head -2
    echo -n "ours:   "; gpurun $O/own_host $O/$k.co $k $shape 30 256 256 128 2>&1 | grep -E '^\{"host"|FAIL|HANG|mismatch' | head -3
  done
done
