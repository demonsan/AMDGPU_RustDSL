#!/usr/bin/env bash
# PMC counter passes on e08 (MI450 / gfx1250) - per-dispatch counters and kernel traces with rocprofv3 1.4.1 from the pinned ROCm 10.2
# docker image (the only working PMC stack on e08; the host's /opt/rocm-10.2.0a20261013 rocprofv3 1.4.2 is NOT used).
#
#   pmc_run.sh <tag> <gpu> "<COUNTER COUNTER ...>" <app> [app args...]     one counter pass
#   pmc_run.sh <tag> <gpu> trace <app> [app args...]                        kernel trace only
#   pmc_run.sh <tag> <gpu> list                                              the counters rocprofv3 offers on that GPU
#
# Rules (the user's): every run holds the WHOLE-NODE gpu-lock (no other job on any GPU during the capture), short hard timeouts, NO SPM
# (streaming / VMP arming options are refused), the container is --rm with only /data/blyu mounted, /dev/kfd + /dev/dri, and the GPU is
# passed explicitly (HIP_VISIBLE_DEVICES). The image's ROCm lives under /root, so the pass runs as the container's root; its output is handed
# back to the calling user (chown) before the container exits. Output: CSV under $PMC_ROOT/<tag>/<pass>/ (default /data/blyu/amdrustdsl/pmc).
set -euo pipefail
IMAGE="rocm/atom-dev:ub24-dcgpu-rk-10.2.0a20260922.bkc.20261002-pyt2.14.0-tri-main-atomsrc-gfx1250-1009"
PMC_ROOT="${PMC_ROOT:-/data/blyu/amdrustdsl/pmc}"
PASS_TIMEOUT="${PASS_TIMEOUT:-120}"
[ "$#" -ge 3 ] || { sed -n '2,12p' "$0"; exit 2; }
TAG="$1"; GPU="$2"; WHAT="$3"; shift 3
case "$TAG" in *[!A-Za-z0-9._-]*|"") echo "pmc_run: bad tag '$TAG'"; exit 2;; esac
case "$GPU" in [0-9]|[0-9][0-9]) ;; *) echo "pmc_run: the GPU must be one physical id"; exit 2;; esac
# no SPM, no streaming, no VMP: refuse any such option anywhere in the request
if printf '%s\n' "$WHAT" "$@" | grep -qiE 'spm|--pmc-stream|vmp|pc[-_]sampl|--att|thread[-_]trace'; then
  echo "pmc_run: REFUSED - SPM / streaming / thread-trace / VMP options are not allowed (counter passes and kernel traces only)"; exit 3
fi
OUT="$PMC_ROOT/$TAG"; mkdir -p "$OUT"
N=$( (ls -d "$OUT"/pass_* 2>/dev/null || true) | wc -l); P="$OUT/pass_$(printf %03d "$N")"; mkdir -p "$P"
case "$WHAT" in
  list)  PROF="/opt/rocm/bin/rocprofv3 --list-avail" ; APP="" ;;
  trace) PROF="/opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d $P -o run --" ; APP="$*" ;;
  *)     # one pass: the counters as given (rocprofv3 refuses a set the hardware cannot count in one pass)
         for c in $WHAT; do case "$c" in *[!A-Za-z0-9_:\[\]=.-]*) echo "pmc_run: bad counter '$c'"; exit 2;; esac; done
         PROF="/opt/rocm/bin/rocprofv3 --pmc $WHAT --output-format csv -d $P -o run --" ; APP="$*" ;;
esac
[ "$WHAT" = list ] || [ -n "$APP" ] || { echo "pmc_run: no application"; exit 2; }
echo "pmc_run: tag=$TAG gpu=$GPU pass=$(basename "$P") what='$WHAT'"
echo "$WHAT | $APP" > "$P/request.txt"
# the whole node, exclusive; the container in the foreground (never -d); a hard kill inside AND outside
gpu-lock timeout -s KILL $((PASS_TIMEOUT + 60)) docker run --rm --network none \
  --device /dev/kfd --device /dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU" -e PROBE_WATCHDOG_S="${PROBE_WATCHDOG_S:-20}" -e CHECK_B -e CHECK_ROWS -e CLUSTER -e STAMPS -e PERSIST \
  -v /data/blyu:/data/blyu -w "$PWD" --entrypoint /bin/bash "$IMAGE" \
  -c "ulimit -f 2097152; timeout -s KILL $PASS_TIMEOUT $PROF $APP; r=\$?; chown -R $(id -u):$(id -g) $P; exit \$r" 2>&1 | grep -vE '^\s*$' | tee "$P/log.txt" | tail -n 40"
rc=${PIPESTATUS[0]}
du -sh "$P" | sed 's/^/pmc_run: output /'
ls "$P" | sed 's/^/  /' | head -20
exit $rc
