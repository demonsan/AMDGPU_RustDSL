# probe_g1250_permpk16 on the GPU, gated (a mistake here can hang the device): the emitted text is checked first (entry prologue, a safe
# follower or v_nop after every v_perm_pk16_*, no EXEC write, G5 clean), then WGS=1 with a 5 s watchdog; WGS=64 only if that was clean.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/permpk16; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
k=probe_g1250_permpk16
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/emit.log 2>&1 || { tail -3 $O/emit.log; echo STOP-EMIT; exit 4; }
echo "--- checks"
head -20 $O/$k.s | grep -nE "s_mov_b64 s\[[0-9]+:[0-9]+\], 0|^\s+v_nop|global_prefetch_b8 v0" | head -3
grep -nE "v_perm_pk16" -A1 $O/$k.s
nexec=$(grep -cE "exec|v_cmpx" $O/$k.s); echo "EXEC mentions: $nexec"
[ "$nexec" = "0" ] || { echo STOP-EXEC; exit 3; }
python3 - "$O/$k.s" <<'EOF' || { echo STOP-FOLLOWER; exit 3; }
import sys, re
L = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.strip().endswith(':') and not l.strip().startswith(('.', '//'))]
safe = re.compile(r'^(v_nop|v_mov_b32|v_add_f32|v_sub_f32|v_mul_f32|v_fma_f32|v_add_nc_u32|v_and_b32|v_or_b32|v_xor_b32|v_lshlrev_b32|v_lshrrev_b32|v_cndmask_b32|v_exp_f32|v_wmma_)(_e32|_e64)?\b')
ok = True
for i, l in enumerate(L):
    if l.startswith('v_perm_pk16'):
        n = L[i + 1] if i + 1 < len(L) else ''
        good = bool(safe.match(n)) and '::' not in n and '_dpp' not in n
        print(('OK  ' if good else 'BAD ') + l + '  ->  ' + n)
        ok &= good
pro = L[:6]
p = any(a.startswith('s_mov_b64') and a.endswith(', 0') and b == 'v_nop' and c.startswith('global_prefetch_b8 v0,') for a, b, c in zip(pro, pro[1:], pro[2:]))
print('prologue', 'OK' if p else 'MISSING')
sys.exit(0 if ok and p else 1)
EOF
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || { echo STOP-ASM; exit 5; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
echo "--- GPU: $IDLE_GPU"
export PROBE_WATCHDOG_S=5
out=$(WGS=1 gpurun $O/host permpk16 $O/$k.co $k 2>&1); rc=$?
echo "$out" | grep -vE "gpu-lock" | tail -8; echo "rc=$rc"
echo "$out" | grep -q "PROBE_OK" && [ $rc = 0 ] || { echo "STOP-ANOMALY (GPU $IDLE_GPU)"; exit 3; }
out=$(WGS=64 gpurun $O/host permpk16 $O/$k.co $k 2>&1); rc=$?
echo "$out" | grep -E "permpk16:|HANG|error" | tail -3; echo "rc=$rc"
echo "$out" | grep -q "PROBE_OK" && [ $rc = 0 ] || { echo "STOP-ANOMALY-64 (GPU $IDLE_GPU)"; exit 3; }
echo PERMPK16-PROBE-DONE
