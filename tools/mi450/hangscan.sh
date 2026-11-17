# the hang-safety asm rules over the disassembly of every vendor code object in $BASE/vendor/gfx1250 (runtime-only content: nothing of it leaves the node's work dir, nothing is copied into the repo)
cd "$SRC"
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"
OD=/opt/rocm/llvm/bin/llvm-objdump
O="$W/hangscan"; rm -rf "$O"; mkdir -p "$O"
n=0
for co in $(find $BASE/vendor/gfx1250 -name '*.co' | sort); do
  b=$(basename $co .co)
  $OD -d --mcpu=gfx1250 $co > $O/$b.s 2> $O/$b.err || echo "objdump failed: $b"
  n=$((n+1))
done
echo "disassembled $n code objects; total $(cat $O/*.s | wc -l) lines"
$A hang-scan --show ${SHOW:-3} $O/*.s > $O/scan.txt
echo "scan exit=$?"
echo "files clean: $(grep -c ': clean' $O/scan.txt), with violations: $(grep -c 'violation(s)' $O/scan.txt)"
echo "--- violations per rule (sum over files)"
awk '/^  GV-1250/ {gsub(":","",$1); c[$1]+=$2} END {for (k in c) print c[k], k}' $O/scan.txt | sort -rn
echo "--- files with violations"
grep 'violation(s)' $O/scan.txt | sed 's#.*/##' | head -40
echo "--- sample lines"
grep -A3 -E '^  GV-1250' $O/scan.txt | head -60
