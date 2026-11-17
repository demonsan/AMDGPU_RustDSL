# perf_ab (tools/perf_ab.py) on the MI450 node: same-session A/B of two commits for the gfx1250 case library (+ the vendor side). Needs `-Sync -Bundle` (git history for the worktrees).
#   env: OLD (default HEAD), NEW (default .), FAMILIES, ONLY, REPS (3), EXTRA (more perf_ab args), OUT (default $W/ab/run)
cd "$SRC"
export PERF_AB_ARCH=gfx1250
export AITER_HSA=${AITER_HSA:-$BASE/vendor/gfx1250}
if [ -z "$IDLE_GPU" ]; then echo "NO-IDLE-GPU"; exit 251; fi
export HIP_VISIBLE_DEVICES=$IDLE_GPU
python3 tools/perf_ab.py --arch gfx1250 --old "${OLD:-HEAD}" --new "${NEW:-.}" --release --reps "${REPS:-3}" ${FAMILIES:+--families $FAMILIES} ${ONLY:+--only "$ONLY"} --out "${OUT:-$W/ab/run}" ${EXTRA:-} 2>&1 | tail -${TAIL:-60}
