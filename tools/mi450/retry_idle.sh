# Runs another payload script ($1), retrying while gpurun finds no idle GPU (rc 251): every 5 min, at most $2 tries (default 16).
set -u
n=${2:-16}
for i in $(seq 1 $n); do
  out=$(IDLE_GPU=${IDLE_GPU:-} bash "$SRC/$1" 2>&1); rc=$?
  if ! echo "$out" | grep -q "rc=251"; then echo "$out"; exit $rc; fi
  echo "try $i: no idle GPU $(date +%H:%M)"
  sleep 300
done
echo "GAVE-UP: no idle GPU"