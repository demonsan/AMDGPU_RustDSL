# Rust toolchain in the user directory ($BASE/rustup, $BASE/cargo): no root, minimal profile. Pinned to the toolchain of tools/container/Containerfile (1.83.0), which the tests assume
# (the classic target/<profile>/deps layout; newer cargo versions lay the test binaries out differently). Idempotent.
set -e
TC=${TC:-1.83.0}
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
  curl -sSf -m 120 https://sh.rustup.rs -o "$BASE/rustup-init.sh"
  sh "$BASE/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain "$TC" 2>&1 | tail -3
  rm -f "$BASE/rustup-init.sh"
fi
rustup toolchain install "$TC" --profile minimal 2>&1 | tail -2
rustup default "$TC" 2>&1 | tail -1
for t in $(rustup toolchain list | awk '{print $1}' | grep -v "^$TC"); do rustup toolchain uninstall "$t" 2>&1 | tail -1; done
echo "$(cargo --version) / $(rustc --version)"
df -h "$BASE" | tail -1
du -sh "$CARGO_HOME" "$RUSTUP_HOME" 2>/dev/null
