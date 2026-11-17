# Run a bash payload on an MI450 (gfx1250) node: optional repo sync, idle-GPU selection, gpu-lock on a08-1, hard timeouts, hang watchdog.
# ASCII only (Windows PowerShell 5.1 and pwsh).
#
#   mi450-exec.ps1 -Node f01|a08 -Payload '<bash>' [-PayloadFile file.sh] [-Sync] [-TimeoutSec 600] [-GpuIdx N] [-NoGpu]
#
# What the payload sees (exported by the bootstrap):
#   $W          work dir on the node   (f01-2: /data/blyu/amdrustdsl, a08-1: ~/amdrustdsl);  $SRC = $W/src (the synced repo), $CARGO_TARGET_DIR = $W/target
#   $IDLE_GPU   index of an idle GPU picked by the bootstrap (amd-smi: no process, GFX activity low, VRAM nearly empty), or empty when none is idle / -NoGpu
#   gpurun CMD  runs CMD on that GPU with HIP_VISIBLE_DEVICES=$IDLE_GPU and a hard `timeout -s KILL $GPU_TIMEOUT`; on a08-1 it goes through /usr/local/bin/gpu-lock
#               (the lock is taken exactly once, at this leaf: never nest gpurun, never lock builds). On a hang (exit 137 / 124) it prints a HANG line and exits: do NOT retry blindly.
# Hard limits: the whole payload runs under `timeout -s KILL <TimeoutSec>` (its own process group, so children die too); gpurun has its own limit (-GpuTimeoutSec, default 120).
# Output: only what the payload prints (the node's login banner is filtered); the exit code of the payload is the exit code of this script (250 = ssh / sync failure, 251 = no idle GPU).
param(
    [Parameter(Mandatory = $true)][ValidateSet('f01', 'a08')][string]$Node,
    [string]$Payload = '',
    [string]$PayloadFile = '',
    [switch]$Sync,
    [string[]]$Env = @(),   # extra environment of the payload: 'K=V' (single quotes in V are not allowed)
    [switch]$Bundle,   # with -Sync: also ship the git history (a bundle of HEAD and origin/Main, about 8 MB) so $SRC is a git repo: perf_ab-style A/B runs can add worktrees of other commits
    [int]$TimeoutSec = 600,
    [int]$GpuTimeoutSec = 120,
    [int]$GpuIdx = -1,
    [switch]$NoGpu,
    [string]$Repo = '',
    [string]$Tag = 'sched'   # session name: the work dir (src, target, logs) is $BASE/ws/<Tag>; the toolchain (cargo, rustup) is shared at $BASE. Two sessions must use two tags (one account on the node).
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Repo) { $Repo = (Get-Location).Path }
$cfg = @{
    f01 = @{ host = '10.5.236.36'; work = '/data/blyu/amdrustdsl'; lock = $false }
    a08 = @{ host = 'ctheliosr-1b112-a08-1.mnb.dcgpu'; work = '$HOME/amdrustdsl'; lock = $true }
}[$Node]
$key = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$kh = Join-Path $env:USERPROFILE '.ssh\mi450_known_hosts'
$wsdir = "$($cfg.work)/ws/$Tag"
$sshBase = @('-i', $key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=20', '-o', 'StrictHostKeyChecking=accept-new', '-o', "UserKnownHostsFile=$kh", "blyu@$($cfg.host)")

function Invoke-Remote([string]$bash, [switch]$Wrap) {
    if ($Wrap) { $bash = "echo @@MI450-BEGIN@@`n" + $bash + "`n" + 'echo "@@MI450-END rc=$?@@"' }
    # base64 transport: no quoting problems, no ';' on the ssh command line
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($bash -replace "`r`n", "`n")))
    $remote = "echo $b64 | base64 -d | bash -l"
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $lines = & ssh @sshBase $remote 2>&1
    $ErrorActionPreference = $eap
    $rc = $LASTEXITCODE
    if ($env:MI450_RAW) { $lines | ForEach-Object { Write-Host "RAW| $_" } }
    $inside = $false
    $exit = $null
    foreach ($l in $lines) {
        $s = "$l"
        if ($s -match '^@@MI450-END rc=(\d+)@@') { $exit = [int]$Matches[1]; $inside = $false; continue }
        if ($s -eq '@@MI450-BEGIN@@') { $inside = $true; continue }
        if ($inside) { Write-Host $s }
    }
    if ($null -eq $exit) { Write-Error "no end marker (ssh rc=$rc): the connection or the node failed" -ErrorAction Continue; return 250 }
    return $exit
}

if ($Sync) {
    # tracked + untracked-not-ignored files of the working tree (uncommitted edits included), streamed as one tar to $W/src
    Push-Location $Repo
    $list = Join-Path $env:TEMP ("mi450-files-" + [guid]::NewGuid().ToString('N') + '.txt')
    $tgz = Join-Path $env:TEMP ("mi450-src-" + [guid]::NewGuid().ToString('N') + '.tgz')
    (git ls-files --cached --others --exclude-standard) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Set-Content -Encoding ASCII $list
    & tar -czf $tgz -T $list
    Pop-Location
    $sz = [math]::Round((Get-Item $tgz).Length / 1MB, 1)
    Write-Host "[mi450] syncing $sz MB to $Node"
    $rcMk = Invoke-Remote ("mkdir -p " + $wsdir.Replace('$HOME', '~')) -Wrap
    $scpArgs = @('-i', $key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', "UserKnownHostsFile=$kh")
    $eap2 = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & scp @scpArgs $tgz ("blyu@" + $cfg.host + ":" + $wsdir.Replace('$HOME', '~') + "/src.tgz") 2>&1 | Out-Null
    $scprc = $LASTEXITCODE; $ErrorActionPreference = $eap2
    if ($Bundle) {
        Push-Location $Repo; $bun = Join-Path $env:TEMP ("mi450-" + [guid]::NewGuid().ToString('N') + '.bundle'); $headSha = (git rev-parse HEAD).Trim(); & git bundle create $bun HEAD origin/Main 2>&1 | Out-Null; Pop-Location
        $ErrorActionPreference = 'Continue'; & scp @scpArgs $bun ("blyu@" + $cfg.host + ":" + $wsdir.Replace('$HOME', '~') + "/src.bundle") 2>&1 | Out-Null; $ErrorActionPreference = $eap2; Remove-Item $bun -ErrorAction SilentlyContinue
    }
    if ($scprc -ne 0) { Write-Error "scp failed ($scprc)"; exit 250 }
    $ok = Invoke-Remote ("W=$wsdir; cd `$W && rm -rf src.new src.old && mkdir src.new && tar -xzf src.tgz -C src.new && { find src.new -type f \( -name '*.sh' -o -name '*.py' -o -name '*.rs' -o -name '*.toml' -o -name '*.md' -o -name '*.txt' -o -name '*.lock' -o -name '*.hip' -o -name '*.cpp' -o -name '*.h' -o -name '*.asm' -o -name '*.tbl' -o -name '*.json' -o -name '*.yml' -o -name '*.tsv' -o -name '.gitignore' -o -name '.gitattributes' -o -name '*.ps1' -o -name '*.irv1' -o -name '*.s' \) -print0 | xargs -0 -r sed -i 's/\r$//'; [ -d src ] && mv src src.old; mv src.new src; rm -rf src.old src.tgz; echo SYNCED; }") -Wrap
    if ($Bundle) {
        $okb = Invoke-Remote ("W=$wsdir; cd `$W/src && rm -rf .git && git init -q && git fetch -q ../src.bundle '+refs/remotes/origin/*:refs/remotes/origin/*' HEAD:refs/b/head && git update-ref --no-deref HEAD `$(git rev-parse refs/b/head) && git reset -q && rm -f ../src.bundle && echo BUNDLED `$(git rev-parse --short HEAD)") -Wrap
        if ($okb -ne 0) { Write-Error 'bundle step failed'; exit 250 }
    }
    Remove-Item $list, $tgz -ErrorAction SilentlyContinue
    if ($ok -ne 0) { Write-Error 'sync failed'; exit 250 }
}

if ($PayloadFile) { $Payload = Get-Content -Raw -LiteralPath $PayloadFile }
if (-not $Payload.Trim()) { if ($Sync) { exit 0 } else { Write-Error 'no payload'; exit 2 } }

$lockCmd = if ($cfg.lock) { '/usr/local/bin/gpu-lock ' } else { '' }
$noGpuFlag = if ($NoGpu) { 1 } else { 0 }
$envLines = ($Env | ForEach-Object { $kv = $_ -split '=', 2; "export $($kv[0])='$($kv[1])'" }) -join "`n"
$boot = @"
export BASE=$($cfg.work)
export W=`$BASE/ws/$Tag
mkdir -p `$W
export SRC=`$W/src CARGO_TARGET_DIR=`$W/target
export CARGO_HOME=`$BASE/cargo RUSTUP_HOME=`$BASE/rustup
export PATH=`$CARGO_HOME/bin:/opt/rocm/bin:`$PATH
export GPU_TIMEOUT=$GpuTimeoutSec
IDLE_GPU=''
if [ $noGpuFlag = 0 ]; then
  if [ $GpuIdx -ge 0 ]; then IDLE_GPU=$GpuIdx; else
    # idle = no process on it, VRAM < 2 GB used, GFX activity < 5 % (amd-smi); the first such GPU
    IDLE_GPU=`$(amd-smi metric -g all --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
d = d.get("gpu_data", d) if isinstance(d, dict) else d
def num(x):
    if isinstance(x, dict): x = x.get("value", 0)
    try: return float(x)
    except Exception: return 0.0
for g in d:
    idx = g.get("gpu", None)
    act = num(g.get("usage", {}).get("gfx_activity", 0))
    used = num(g.get("mem_usage", {}).get("used_vram", 0))
    if idx is not None and act < 5 and used < 2048:
        print(idx); break
' 2>/dev/null)
  fi
fi
export IDLE_GPU
if [ -n "`$IDLE_GPU" ]; then export HIP_VISIBLE_DEVICES=`$IDLE_GPU; fi
export PERF_AB_GPU_WRAP="${lockCmd}timeout -s KILL `$GPU_TIMEOUT"
$envLines
gpurun() {
  if [ -z "`$IDLE_GPU" ]; then echo "NO-IDLE-GPU: refusing to run `$*" >&2; return 251; fi
  HIP_VISIBLE_DEVICES=`$IDLE_GPU ${lockCmd}timeout -s KILL `$GPU_TIMEOUT "`$@"
  local rc=`$?
  if [ `$rc -eq 137 ] || [ `$rc -eq 124 ]; then echo "HANG: `$* exceeded `${GPU_TIMEOUT}s on GPU `$IDLE_GPU (killed). Stop, report, do not retry blindly." >&2; fi
  return `$rc
}
export -f gpurun
PF=`$(mktemp `$W/.payload.XXXXXXXX)
cat > `$PF <<'MI450_PAYLOAD_EOF'
$Payload
MI450_PAYLOAD_EOF
echo @@MI450-BEGIN@@
timeout -s KILL $TimeoutSec bash `$PF
rc=`$?
rm -f `$PF
echo "@@MI450-END rc=`$rc@@"
"@
$rc = Invoke-Remote $boot
exit $rc
