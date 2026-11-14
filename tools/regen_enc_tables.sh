#!/usr/bin/env bash
# Regenerate the opcode / format table of the in-process assembler (crates/amdgpu-target/src/gfx950_co/tables.rs) after a new mnemonic appears in the emitted assembly.
# Needs the LLVM tools (AMD_LLVM_BIN). The field layouts per format are code (gfx950_co/enc.rs); `amdc co-check /tmp/oracle.tsv` and the co_oracle test say what is missing.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="${AMD_CARGO_TARGET:-/tmp/target-compose}"
AMD_CARGO_TARGET="$T" bash tools/container/cargo.sh build --release --bin amdc
rm -rf /tmp/allasm && mkdir -p /tmp/allasm
"$T/release/amdc" compile-batch --jobs 16 --asm-dir /tmp/allasm >/dev/null 2>&1
python3 tools/mk_enc_oracle.py /tmp/allasm /tmp/oracle.tsv
python3 tools/gen_enc_tables.py            # writes /tmp/enc_tables.rs
cp /tmp/enc_tables.rs crates/amdgpu-target/src/gfx950_co/tables.rs
"$T/release/amdc" co-check /tmp/oracle.tsv