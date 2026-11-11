#!/usr/bin/env bash
# GPU guard, SOURCED by every script that runs GPU work (USER RULE 2026-11-11: GPU tests run only on GPUs 0-3, never on 4-7, on any node).
#
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"      (after the script's own `export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"`)
#
# Requires HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES) to be set and to list only GPUs 0-3, else prints why and exits 1 (the sourcing script stops).
# Both variables are checked when both are set. A device list that is not plain numbers (UUIDs) is refused too: it cannot be proven to be 0-3.
_gpu_guard_check() {
    local name v g any=0
    for name in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
        v="${!name:-}"
        [ -n "$v" ] || continue
        any=1
        for g in ${v//,/ }; do
            case "$g" in
                0|1|2|3) ;;
                *) echo "gpu_guard: GPU '$g' in $name='$v' is not allowed: GPU tests may only run on GPUs 0-3 (never 4-7, on any node)" >&2; return 1 ;;
            esac
        done
    done
    if [ "$any" = 0 ]; then
        echo "gpu_guard: HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES) is not set: set it to a GPU in 0-3 (e.g. HIP_VISIBLE_DEVICES=0)" >&2
        return 1
    fi
}
_gpu_guard_check || exit 1