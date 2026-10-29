#!/usr/bin/env bash
# Structural analysis of the vendor FMHA code object (and of our kernel) - the evidence behind
# docs/vendor-spec.md. Writes disassembly only under build/ (gitignored); the vendor .co itself is
# never copied into the repository.
#
# Usage:
#   bash tools/vendor_disasm.sh survey                 # vendor: summary + labels + loops + sync map
#   bash tools/vendor_disasm.sh phases <label> --seq   # vendor: barrier phases of a named loop
#   bash tools/vendor_disasm.sh region <label>         # vendor: opcode histogram + RLE sequence
#   bash tools/vendor_disasm.sh mfma|lds|staging|branches
#   bash tools/vendor_disasm.sh summary <objdump.txt>  # any dump (e.g. ours)
#   bash tools/vendor_disasm.sh objdump <code.co>      # only disassemble, print the dump path
#
# Env: REF_CO (vendor .co, default the shared session copy), ROCM_PATH (needs llvm-objdump with the
# gfx950 backend; default ROCm 7.2.4), VENDOR_WORK (default build/vendor).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
OD="$ROC/llvm/bin/llvm-objdump"
OUT="${VENDOR_WORK:-$REPO/build/vendor}"
REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
mkdir -p "$OUT"
cd "$REPO"

disasm() { # $1 = .co, $2 = dump path
    "$OD" -d --mcpu=gfx950 --no-show-raw-insn "$1" > "$2" 2>/dev/null
    [ -s "$2" ] || { echo "disassembly failed: $1" >&2; return 1; }
    echo "$2"
}

vendor_dump() {
    if [ ! -s "$OUT/vendor.objdump.txt" ] || [ "$REF_CO" -nt "$OUT/vendor.objdump.txt" ]; then
        disasm "$REF_CO" "$OUT/vendor.objdump.txt" >/dev/null || exit 1
    fi
    echo "$OUT/vendor.objdump.txt"
}

cmd="${1:-survey}"
shift || true
case "$cmd" in
    survey)
        D=$(vendor_dump)
        python3 tools/vendor_analyze.py "$D" summary
        echo
        python3 tools/vendor_analyze.py "$D" labels
        echo
        python3 tools/vendor_analyze.py "$D" loops --top 8
        echo
        python3 tools/vendor_analyze.py "$D" sync
        ;;
    phases|region|loops|labels|mfma|lds|staging|sync|branches|summary)
        D=$(vendor_dump)
        python3 tools/vendor_analyze.py "$D" "$cmd" "$@"
        ;;
    objdump)
        CO="${1:?usage: vendor_disasm.sh objdump <code.co>}"
        P="$OUT/$(basename "${CO%.*}").objdump.txt"
        disasm "$CO" "$P"
        ;;
    analyze)
        F="${1:?usage: vendor_disasm.sh analyze <objdump.txt> [command...]}"
        shift || true
        python3 tools/vendor_analyze.py "$F" "${@:-summary}"
        ;;
    *)
        echo "usage: bash tools/vendor_disasm.sh [survey|phases|region|loops|labels|mfma|lds|staging|sync|branches|objdump|analyze]" >&2
        exit 2
        ;;
esac
