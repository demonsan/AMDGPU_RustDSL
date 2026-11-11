#!/usr/bin/env bash
# GPU guard, SOURCED by every script that runs GPU work (USER RULE 2026-11-11: GPU tests run only on GPUs 0-3, never on 4-7, on any node).
#
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"      (after the script's own `export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"`)
#
# Requires HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES) to be set and the PHYSICAL GPUs it selects to be 0-3, else prints why and exits 1 (the sourcing script stops).
# HIP_VISIBLE_DEVICES indexes into the set ROCR_VISIBLE_DEVICES leaves visible (a scheduler job may export ROCR_VISIBLE_DEVICES=0,...,7 and the script then picks HIP_VISIBLE_DEVICES=1:
# that is physical GPU 1). A device list that is not plain numbers (UUIDs) is refused: it cannot be proven to be 0-3.
_gpu_guard_check() {
    local hip="${HIP_VISIBLE_DEVICES:-}" rocr="${ROCR_VISIBLE_DEVICES:-}" g i phys
    local -a rl hl
    if [ -z "$hip" ] && [ -z "$rocr" ]; then
        echo "gpu_guard: HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES) is not set: set it to a GPU in 0-3 (e.g. HIP_VISIBLE_DEVICES=0)" >&2
        return 1
    fi
    IFS=',' read -r -a rl <<< "$rocr"
    IFS=',' read -r -a hl <<< "$hip"
    for g in "${rl[@]}"; do
        case "$g" in ''|*[!0-9]*) echo "gpu_guard: ROCR_VISIBLE_DEVICES='$rocr' is not a plain device-number list: cannot prove it is 0-3" >&2; return 1 ;; esac
    done
    if [ -z "$hip" ]; then
        for g in "${rl[@]}"; do
            case "$g" in 0|1|2|3) ;; *) echo "gpu_guard: GPU '$g' in ROCR_VISIBLE_DEVICES='$rocr' is not allowed: GPU tests may only run on GPUs 0-3 (never 4-7, on any node)" >&2; return 1 ;; esac
        done
        return 0
    fi
    for i in "${hl[@]}"; do
        case "$i" in ''|*[!0-9]*) echo "gpu_guard: HIP_VISIBLE_DEVICES='$hip' is not a plain device-number list: cannot prove it is 0-3" >&2; return 1 ;; esac
        if [ -n "$rocr" ]; then
            if [ "$i" -ge "${#rl[@]}" ]; then
                echo "gpu_guard: HIP_VISIBLE_DEVICES='$hip' indexes into ROCR_VISIBLE_DEVICES='$rocr' (${#rl[@]} entries): index $i is out of range, every launch would fail. Set only one of the two variables" >&2
                return 1
            fi
            phys="${rl[$i]}"
        else phys="$i"; fi
        case "$phys" in
            0|1|2|3) ;;
            *) echo "gpu_guard: HIP_VISIBLE_DEVICES='$hip' selects physical GPU '${phys:-?}'${rocr:+ (ROCR_VISIBLE_DEVICES='$rocr')}: GPU tests may only run on GPUs 0-3 (never 4-7, on any node)" >&2; return 1 ;;
        esac
    done
}
_gpu_guard_check || exit 1
