# MI450 (gfx1250) test nodes: access, toolchain, GPU runs, hang policy

Two shared 4-GPU nodes carry gfx1250 GPUs (wave 32, 256 CUs each) and the ROCm clang 24 toolchain at `/opt/rocm`. This page is the operating manual for running our build, tests and GPU code on them. It contains infrastructure rules only.

| | f01-2 | a08-1 |
|---|---|---|
| ssh | `blyu@10.5.236.36` (the host name does not resolve from the workstation) | `blyu@ctheliosr-1b112-a08-1.mnb.dcgpu` |
| GPU lock | none installed (cooperate by checking `amd-smi`) | **mandatory**: every GPU command through `/usr/local/bin/gpu-lock` |
| big disk | `/data/blyu` (`/data` is about 98 % full: keep the footprint small, delete build trees) | the home directory (`/` is about 96 % full) |
| CPUs / RAM | 256 / 503 GB | 256 |

Both nodes are shared with other teams and with the other sessions of this project (one account, `blyu`). Nobody owns a GPU: pick an idle one every time and never touch a process you did not start.

## 0. e08-01 (the third node)

| | e08-01 |
|---|---|
| login | `amd@10.223.206.118`, a **shared** account; **password login only**: never install an ssh key or set up passwordless login there (user policy). `mi450-exec.ps1 -Node e08` logs in through a local askpass helper kept outside the repository; use only `mi450-exec` for this node (no direct `ssh` / `scp`) |
| work dir | `/data/blyu/amdrustdsl` (`$BASE`; sessions under `$BASE/ws/<Tag>`). The account is shared with other people: touch nothing outside `/data/blyu` |
| GPU lock | per-GPU `gpu-lock` (`pergpu`: `gpurun` locks only `IDLE_GPU`, `gpu-lock -g <n>`); same rules as on a08-1 |
| host compiler | Ubuntu 26.04: gcc 16 ships no libstdc++ headers, so the HIP hosts are compiled against gcc 15. The runner exports `HIPCC_COMPILE_FLAGS_APPEND` / `CCC_OVERRIDE_OPTIONS=+--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15`; the `### CCC_OVERRIDE_OPTIONS` / `unused argument` lines in the build output are expected |
| vendor code objects | copied to `$BASE/vendor/gfx1250/{f4gemm,mla}` (the same files as on a08-1) |
| known test result | `co_oracle` (`the_in_process_code_object_is_byte_identical_to_clangs_for_every_reachable_kernel`) **fails there by design**: the node's clang (ROCm 10.3) is not the pinned assembler, so its code objects differ. Do not chase it; the gate there is the md5 / co / added / cert manifests, the other tests and `lite_contract` as everywhere |
| profiler | the release `rocprofv3` lacks the gfx1250 derived metrics, and most SQ / TX counters read 0 without VMP arming (shared GPU state: needs the user's authorisation; see the mi450-spm-capture skill). Do not arm VMP or install a profiler runtime without that authorisation |

## 1. `_scripts/mi450-exec.ps1` (`tools/mi450/mi450-exec.ps1`)

```
mi450-exec.ps1 -Node f01|a08|e08 [-Sync -Repo <worktree>] [-PayloadFile x.sh | -Payload '<bash>'] [-Tag <session>]
               [-TimeoutSec 600] [-GpuTimeoutSec 120] [-GpuIdx N] [-NoGpu]
```

- **Sync** (`-Sync`): the tracked and untracked-but-not-ignored files of the worktree (uncommitted edits included) go over as one `tar` (about 3 MB) to `$W/src`; no history, no `.git`. A sync only replaces `$W/src`; build output survives.
- **Payload**: a bash script run on the node by a login shell (transport: base64, so there is no quoting problem; keep `;`-heavy one-liners in a file). The node's login banner is filtered; the exit code of the payload is the exit code of the script (250: ssh / sync failure, 251: no idle GPU).
- **Sessions** (`-Tag`, default `sched`): the work directory is `$BASE/ws/<Tag>` (`$BASE` = `/data/blyu/amdrustdsl` on f01-2, `~/amdrustdsl` on a08-1). It holds `src/` (the synced tree), `target/` (the cargo target dir, kept between runs) and the logs. **Two sessions must use two tags**: the account is shared, and a shared fixed path was already the cause of one session executing another's payload (payload files are now `mktemp` files).
- **Environment of the payload**: `BASE`, `W` (work dir), `SRC` (`$W/src`), `CARGO_TARGET_DIR` (`$W/target`), `CARGO_HOME` / `RUSTUP_HOME` (shared, `$BASE/cargo`, `$BASE/rustup`), `PATH` (cargo and `/opt/rocm/bin`), `IDLE_GPU`, `GPU_TIMEOUT`, and the function `gpurun`.
- **Hard timeouts**: the whole payload runs under `timeout -s KILL <TimeoutSec>`; `timeout` puts the command in its own process group and kills the group, so child processes die with it.

## 2. Picking and using a GPU

- **Idle test** (done by the bootstrap, once per run; use `-GpuIdx` to force a GPU): `amd-smi metric` says `gfx_activity` below 5 % and `used_vram` below 2 GB. The first such GPU becomes `IDLE_GPU`. With none idle, `gpurun` refuses (exit 251): wait, do not share a busy GPU. A time-of-check race with another user is possible on f01-2 (no lock there), so check again before a long run (`amd-smi process`).
- **`gpurun CMD ...`** runs `CMD` with `HIP_VISIBLE_DEVICES=$IDLE_GPU` under `timeout -s KILL $GPU_TIMEOUT` (`-GpuTimeoutSec`, default 120 s) and, on a08-1, under `/usr/local/bin/gpu-lock`. Rules, from the lock's own conventions: take the lock **once, at the leaf GPU command** (builds, `cargo`, emitting code objects never take it); never nest `gpurun`; keep the command in the foreground; do not hold the lock while analysing results.
- A GPU command always goes through `gpurun`. Building a code object or running the CPU test suite does not.

## 3. Hang policy (every GPU run has a hard limit)

1. Every GPU run has two limits: the host-side watchdog of the program (the smoke host polls the launch and leaves with exit 124 after `SMOKE_WATCHDOG_S`, default 20 s, **without** touching the device again: no synchronize, free or reset, which can hang too) and `gpurun`'s `timeout -s KILL`.
2. On a timeout `gpurun` prints `HANG: ...` and returns 137 (or 124). **Stop there**: report the case (kernel, shape, node, GPU, what ran before), do not retry blindly, do not run the same kernel on another GPU "to see", and do not try to recover the device (no reset, no killing other users' processes).
3. A kernel that hangs the device can leave the GPU busy for everyone: check `amd-smi` (activity / VRAM) before using that GPU again, and tell the node's owners when it stays busy.
4. A new kernel family or a changed synchronization pattern is first run in the smallest configuration (one workgroup, a few iterations) under a short limit (10 - 20 s), then scaled up.
5. Compile-time rules (the verifier) are the first line of defence against hangs: see the repository's verification rules for the synchronization patterns that are rejected before a kernel ever reaches a GPU.

## 4. Rust toolchain and our build

- **Toolchain**: rustup in the user directory (`$BASE/cargo`, `$BASE/rustup`, no root, minimal profile, about 0.5 GB), **pinned to 1.83.0**, the version of `tools/container/Containerfile` (newer cargo versions lay test binaries out differently and break `amdgpu-safeasm`'s compile-fail test). `tools/mi450/install-rust.sh` installs it idempotently; there is no podman on these nodes, so `tools/container/cargo.sh` is not used.
- **Build and test** (payloads `tools/mi450/build.sh`, `tools/mi450/test.sh`): `mi450-exec.ps1 -Node f01 -NoGpu -Sync -Repo . -PayloadFile tools/mi450/build.sh`, then `test.sh`. Measured on both nodes with `-j 64`: **the release build of the workspace takes about 12 s and `cargo test --release --no-fail-fast` about 85 s: 653 passed, 1 failed, 5 ignored (164 targets)**. The one failure (`lite_contract`) needs the pinned `llc` (`ROCM_LLVM_BIN`), which the nodes do not ship; it is an environment gap, not a regression (the LLVM-tail tests are part of the back-end work for these targets).
- Keep the tree small: `target/` is about 1 GB per session. Delete `$W/target` when a session ends (`rm -rf` of your own `$BASE/ws/<Tag>` only).

## 5. gfx1250 smoke (`tests/gpu/smoke_host.cpp`, `tests/gpu/smoke_kernel.hip`, `tools/mi450/smoke.sh`)

A trivial kernel (`out[i] = 3 i + 1`) is built with the node's ROCm clang as a bare gfx1250 code object, loaded with `hipModuleLoad`, launched through a polled (never blocking) event wait, and checked: `SMOKE_OK` (exit 0) or `SMOKE_FAIL` / `SMOKE_HANG`. Last result on both nodes (GPU 0): `device: AMD Radeon Graphics arch: gfx1250 CUs: 256 wave: 32`, `bad=0 SMOKE_OK`; on a08-1 the run went through `gpu-lock`.
`smoke_host <co> [kernel] [n] [block]` loads any code object whose kernel has the `(int* out, int n)` kernarg shape, so the same host runs the code objects our emitter produces once it targets this architecture. `tools/mi450/watchdog-test.sh` checks the hang path (a command that outlives its limit prints `HANG` and returns 137) on both nodes.

## 6. Quick reference

| task | command |
|---|---|
| sync the worktree and build | `mi450-exec.ps1 -Node f01 -NoGpu -Sync -Repo . -Tag mine -PayloadFile tools/mi450/build.sh` |
| run the test suite | `... -NoGpu -Tag mine -TimeoutSec 3000 -PayloadFile tools/mi450/test.sh` |
| GPU smoke | `... -Tag mine -GpuTimeoutSec 60 -PayloadFile tools/mi450/smoke.sh` |
| which GPU is idle | `... -PayloadFile` with `echo "$IDLE_GPU"` (empty: none idle) |
| clean up | a payload that does `rm -rf "$W/target"` |
