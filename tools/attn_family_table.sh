#!/usr/bin/env bash
# Same-run attention-family ratio table (docs/op-coverage-design.md §7.5 "attention 各族与厂商的当前比值汇总"): one GPU session, every
# prefill / backward family against its aiter gfx950 vendor kernel on the same shapes, ours and the vendor alternating.
#
# Usage (on the node, repo root):  HIP_VISIBLE_DEVICES=0 bash tools/attn_family_table.sh [out_dir]
# Needs the release amdc (AMDC, default /tmp/target-kfmha/release/amdc), clang (CLANG), hipcc (HIPCC) and the aiter hsa tree (AITER_HSA).
# Families: per-tensor FP8 (fp8pt_host), MXFP8 (ref_host, REF_DATA=normal), bf16 v3 hd128 and hd192/hd128 (attn_host), bf16 backward
# (attn_bwd_host). The decode families (paged, MLA) are not in this script: their hosts take a dispatch plan (see §7.5).
set -u
OUT=${1:-/tmp/attn_table}
AMDC=${AMDC:-/tmp/target-kfmha/release/amdc}
CLANG=${CLANG:-/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang}
HIPCC=${HIPCC:-/shared/apps/ubuntu/opt/rocm-7.0.2/bin/hipcc}
HSA=${AITER_HSA:-/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950}
REF_CO=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
mkdir -p "$OUT"
# ONLY=<list of fp8pt,mxfp8,bf16,hd192,bwd>: run only those sections
sec() { [ -z "${ONLY:-}" ] || [[ ",$ONLY," == *",$1,"* ]]; }
emit() { "$AMDC" emit "$1" --backend own --out "$OUT/$1.s" >"$OUT/$1.log" 2>&1 && "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/$1.co" "$OUT/$1.s"; }
vname() { strings "$1" | grep -m1 '^_ZN'; }
for k in fmha_fwd_hd128_fp8pt_v3 fmha_fwd_hd128_fp8pt_v3_causal fmha_fwd_hd128_fp8pt_v3_kvsplit fmha_fwd_hd128_fp8pt_v3_causal_kvsplit fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h fmha_fwd_hd128_mxfp8_v3_vs128 fmha_fwd_hd128_bf16_v3 fmha_fwd_hd128_bf16_v3_ks fmha_fwd_hd128_bf16_v3_causal fmha_fwd_hd128_bf16_v3_causal_ks fmha_fwd_hd128_bf16_v3_causal_q128 fmha_fwd_hd128_bf16_causal \
         fmha_fwd_hd192_hd128_bf16_v3 fmha_fwd_hd192_hd128_bf16_v3_ks fmha_fwd_hd192_hd128_bf16_v3_causal fmha_fwd_hd192_hd128_bf16_v3_causal_np fmha_fwd_hd192_hd128_bf16_v3_causal_np_q128 fmha_fwd_hd192_hd128_bf16_v3_causal_ks attn_combine_bf16_hd128_q16 fmha_bwd_dot_hd128_bf16 fmha_bwd_dq_hd128_bf16 fmha_bwd_dq_hd128_bf16_causal \
         fmha_bwd_dkdv_hd128_bf16 fmha_bwd_dkdv_hd128_bf16_causal; do
    emit "$k" || echo "EMIT FAIL $k"
done
for h in fp8pt_host attn_host attn_bwd_host; do "$HIPCC" -O2 -std=c++17 "tests/gpu/$h.cpp" -o "$OUT/$h" || echo "BUILD FAIL $h"; done
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o "$OUT/ref_host" -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib 2>/dev/null \
    || "$HIPCC" -O2 -std=c++17 tests/gpu/ref_host.cpp -o "$OUT/ref_host"

sec fp8pt && echo "## fp8 per-tensor (SCALAR_DS=1: the descale form the vendor binary honours)"
sec fp8pt && for c in 0 1; do
    s=""; [ $c = 1 ] && s=_causal
    for sh in "2 32 32 8192 8192" "4 16 16 2048 2048" "1 8 8 4096 4096" "2 32 8 4096 4096" "1 8 8 2048 2048"; do
        echo "-- fp8pt causal=$c $sh"
        # the form dispatch::fp8pt_prefill_kernel picks (CUs = 256): KV split S when the unpaired grid U < 256
        set -- $sh; u=$(( $1 * $2 * (($4 + 255) / 256) )); n=1; while [ $n -lt 4 ] && [ $((u * n)) -lt 256 ] && [ $((($5 + 127) / 128)) -ge $((4 * n)) ]; do n=$((n * 2)); done
        k=fmha_fwd_hd128_fp8pt_v3$s; kv=""; [ $n -gt 1 ] && { k=${k}_kvsplit; kv=$n; }
        echo "   ours form: $k${kv:+ S=$kv}"
        env SCALAR_DS=1 CHECK=16 ${kv:+KVS=$kv COMBINE_CO=$OUT/fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h.co} timeout 900 "$OUT/fp8pt_host" "$HSA/fmha_v3_fwd/fwd_hd128_fp8$s.co" "$(vname "$HSA/fmha_v3_fwd/fwd_hd128_fp8$s.co")" \
            "$OUT/$k.co" "$k" $sh $c | grep -E "TF/s|ratio|parity"
    done
done

sec mxfp8 && echo "## MXFP8 (_vs128 forced, REF_DATA=normal)"
sec mxfp8 && for sh in "1 8 8192" "4 8 2048" "2 32 8192"; do
    set -- $sh
    echo "-- mxfp8 B=$1 H=$2 S=$3"
    REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$3 REF_DATA=normal REF_ROWS=4 REF_WARMUP=50 REF_ITERS=50 REF_CO2="$OUT/fmha_fwd_hd128_mxfp8_v3_vs128.co" \
        REF_KNAME2=fmha_fwd_hd128_mxfp8_v3_vs128 REF_ROWS_PER_BLOCK=256 timeout 900 "$OUT/ref_host" "$REF_CO" 2>&1 | grep -E "^timing|vendor kernel vs CPU|REF_KERNEL" | head -5
done

sec bf16 && echo "## bf16 v3 hd128"
sec bf16 && for c in 0 1; do
    s=""; [ $c = 1 ] && s=_causal
    for sh in "2 32 32 8192 8192" "4 16 16 2048 2048" "1 8 8 4096 4096" "4 16 16 1024 1024" "1 32 8 2048 2048" "2 16 16 1024 1024"; do
        echo "-- bf16 causal=$c $sh"
        v="$HSA/fmha_v3_fwd/fwd_hd128_bf16$s.co"
        LSE=0 CHECK=4 timeout 900 "$OUT/attn_host" vendor "$v" "$(vname "$v")" $sh $c | grep -E "TF/s"
        # non-causal: the key split of dispatch::bf16_prefill_key_splits (CUs = 256); causal: the paired kernel (its small-grid `_causal_ks`
        # target needs the backend SGPR fix of sgpr-fix to assemble)
        set -- $sh; u=$(( $1 * $2 * (($4 + 255) / 256) )); n=1; k=fmha_fwd_hd128_bf16_v3$s; ks=""; rows=256; pr1=$c; kt=$((($5 + 127) / 128))
        if [ $c = 0 ]; then while [ $n -lt 4 ] && [ $((u * n)) -lt 256 ] && [ $kt -ge $((4 * n)) ]; do n=$((n * 2)); done
        else p=$(( $1 * $2 * ((($4 + 255) / 256 + 1) / 2) ))
            if [ $p -lt 128 ] && [ $kt -lt 16 ]; then k=fmha_fwd_hd128_bf16_causal; rows=128; pr1=0
            elif [ $p -ge 128 ] && [ $p -lt 256 ] && [ $kt -le 16 ]; then k=${k}_q128; rows=128
            elif [ $kt -ge 16 ]; then while [ $n -lt 4 ] && [ $((p * n)) -lt 256 ] && [ $kt -ge $((4 * n)) ]; do n=$((n * 2)); done; fi
        fi
        [ $n -gt 1 ] && { k=fmha_fwd_hd128_bf16_v3${s}_ks; ks=$(( n == 2 ? 1 : 2 )); }
        echo "   ours form: $k${ks:+ KS=$ks}"
        env LSE=0 CHECK=4 ROWS=$rows BLOCK=$(( rows * 2 )) PAIR=$pr1 ${ks:+KS=$ks COMBINE_CO=$OUT/attn_combine_bf16_hd128_q16.co} timeout 900 "$OUT/attn_host" ours "$OUT/$k.co" "$k" $sh $c | grep -E "TF/s"
    done
done

sec hd192 && echo "## bf16 v3 hd192/hd128 (MLA prefill, vendor 128 rows / 256 threads)"
sec hd192 && for c in 0 1; do
    s=""; [ $c = 1 ] && s=_causal
    v=$(ls "$HSA"/fmha_v3_fwd/fwd_hd192_hd128_bf16$s.co 2>/dev/null | head -1)
    for sh in "1 16 16 4096 4096" "4 16 16 2048 2048" "1 8 8 4096 4096" "1 128 128 1024 1024" "1 16 16 2048 2048"; do
        echo "-- hd192 causal=$c $sh"
        [ -n "$v" ] && ROWS=128 BLOCK=256 DV=128 LSE=0 CHECK=4 timeout 900 "$OUT/attn_host" vendor "$v" "$(vname "$v")" $sh $c 192 | grep -E "TF/s"
        # the form dispatch::hd192_prefill_kernel picks (CUs = 256): paired / unpaired (_np) / unpaired key split (_ks, S = 2)
        set -- $sh; u=$(( $1 * $2 * (($4 + 255) / 256) )); k=fmha_fwd_hd192_hd128_bf16_v3$s; pr=0; ks=""; rows=""
        kt=$((($5 + 127) / 128))
        if [ $c = 1 ]; then
            if [ $u -ge 512 ]; then pr=1; elif [ $((2 * u)) -le 256 ] && { [ $kt -ge 32 ] || { [ $((4 * u)) -le 256 ] && [ $kt -ge 16 ]; }; }; then k=${k}_ks; ks=1; elif [ $((2 * u)) -le 256 ]; then k=${k}_np_q128; rows=128; else k=${k}_np; fi
        elif [ $((4 * u)) -le 256 ] && [ $kt -ge 16 ]; then k=${k}_ks; ks=2
        elif [ $u -lt 256 ] && [ $kt -ge 16 ]; then k=${k}_ks; ks=1
        fi
        echo "   ours form: $k${ks:+ KS=$ks}"
        env DV=128 LSE=0 CHECK=4 ROWS=${rows:-256} BLOCK=$(( ${rows:-256} * 2 )) PAIR=$pr ${ks:+KS=$ks COMBINE_CO=$OUT/attn_combine_bf16_hd128_q16.co COMBINE_KN=attn_combine_bf16_hd128_q16} timeout 900 "$OUT/attn_host" ours "$OUT/$k.co" "$k" $sh $c 192 | grep -E "TF/s"
    done
done

sec bwd && echo "## bf16 backward hd128"
sec bwd && for c in 0 1; do
    for sh in "4 16 16 2048 2048" "1 16 16 4096 4096" "4 16 4 2048 2048"; do
        echo "-- bwd causal=$c $sh"
        VENDOR_DIR="$HSA/fmha_v3_bwd" CHECK=4 timeout 1200 "$OUT/attn_bwd_host" "$OUT" $sh $c 2>&1 | grep -E "bwd .*TF/s|bad" | head -4
    done
done
