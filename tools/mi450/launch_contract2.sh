set -u
cd "$SRC"
O="$W/chk2"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu tests/gpu/g1250_f4own_host.cpp -o $O/h -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib 2>&1 | grep -E "error" | head
k=gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles
$A emit $k --target gfx1250 --backend=own --out $O/$k.s >/dev/null 2>&1; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null
echo "--- a 256 x 256 kernel given BM=128 BN=64"; $O/h $O/$k.co $k 512 512 512 0 128 64 128; echo "rc=$?"
echo "--- ks4 with K = 640"; $O/h $O/$k.co $k 512 512 640 0 256 256 128; echo "rc=$?"
echo "--- legal K = 1024 (1 WG first)"; $O/h $O/$k.co $k 512 512 1024 0 256 256 128 2>&1 | tail -1; echo "rc=$?"