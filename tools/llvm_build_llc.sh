#!/usr/bin/env bash
# Build llc for a ROCm LLVM that ships libLLVM.so but no llc binary (ROCm 10.x), from llvm/tools/llc of the
# SAME llvm-project commit (`clang --version` prints it). No LLVM source is modified; only llc.cpp/NewPMDriver.cpp
# are compiled against the installed headers and linked to the installed libLLVM.so.
#   usage: tools/llvm_build_llc.sh <rocm-root> <llvm-project-checkout> <out-dir>
set -euo pipefail
R="$1"; S="$2/llvm"; T="$3"
mkdir -p "$T/bin" /tmp/llcbuild.$$ && cd /tmp/llcbuild.$$
CFG="$R/llvm/bin/llvm-config"
FL="$($CFG --cxxflags | sed 's/-std=[^ ]*//; s/-fno-exceptions//; s/-fno-rtti//') -std=c++17 -fno-rtti -O1"
g++ $FL -I"$S/tools/llc" -c "$S/tools/llc/llc.cpp" -o llc.o
g++ $FL -I"$S/tools/llc" -c "$S/tools/llc/NewPMDriver.cpp" -o npm.o
g++ llc.o npm.o -o "$T/bin/llc" $($CFG --ldflags) -L"$($CFG --libdir)" -Wl,-rpath,"$($CFG --libdir)" -lLLVM
for f in "$R"/llvm/bin/*; do b=$(basename "$f"); [ "$b" = llc ] || ln -sf "$f" "$T/bin/$b"; done
"$T/bin/llc" --version | head -2