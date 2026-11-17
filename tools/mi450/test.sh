# cargo test --release --no-fail-fast of the synced repo; prints the totals and the failing tests
cd "$SRC"
S=$(date +%s)
cargo test --release --no-fail-fast -j ${JOBS:-64} > "$W/test.log" 2>&1
rc=$?
echo "tests exit=$rc in $(( $(date +%s) - S )) s: $(grep -E 'test result' "$W/test.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed",p,"failed",f,"ignored",i,"targets",NR}')"
grep -E "^test .*FAILED|panicked at" "$W/test.log" | head -${SHOW:-25}
grep -E "^error" "$W/test.log" | head -5
