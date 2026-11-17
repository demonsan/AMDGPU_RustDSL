# cargo build --release of the synced repo ($SRC), target in $CARGO_TARGET_DIR (kept between runs: incremental)
set -e
cd "$SRC"
cargo --version
S=$(date +%s)
cargo build --release -j ${JOBS:-64} 2>&1 | tail -${TAIL:-5}
echo "build: $(( $(date +%s) - S )) s"
ls -la "$CARGO_TARGET_DIR/release/amdc" | cut -c1-80
du -sh "$CARGO_TARGET_DIR" | cut -f1
df -h "$W" | tail -1
