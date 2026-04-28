param(
  [string]$PythonExe = "D:\Junyi_Files\conda-envs\logitc-get\python.exe",
  [string]$ModelPath = "models/Qwen3-8B",
  [string]$CacheDir = "data/sst5/teacher_logits/Qwen3-8B_kspace_seq",
  [string]$OutputDir = "outputs/sst5_kspace_seq_qwen3_lr5e-5",
  [string]$GenerationOutputDir = "",
  [int]$MaxNewTokens = 64,
  [double]$Temperature = 1.0,
  [string]$Device = "cuda",
  [ValidateSet("float16", "bfloat16", "float32")]
  [string]$LoadDtype = "bfloat16",
  [int]$PollSeconds = 30,
  [switch]$TrustRemoteCode,
  [switch]$RunTeacherKBaseline,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

if ($PollSeconds -le 0) {
  throw "PollSeconds must be positive; got $PollSeconds"
}

if ($GenerationOutputDir -eq "") {
  $GenerationOutputDir = Join-Path $OutputDir "latent_generation_eval"
}

$EvalMetrics = Join-Path (Join-Path $OutputDir "test_eval") "eval_metrics.json"
$GenerationMetrics = Join-Path $GenerationOutputDir "generation_metrics.json"
$Checkpoint = Join-Path $OutputDir "best_sst5_content_ae.pt"
$TestCache = Join-Path $CacheDir "test.pt"

function Get-FileLengthOrMinusOne {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return -1
  }
  return (Get-Item $Path).Length
}

function Wait-ForStableFile {
  param(
    [string]$Path,
    [int]$PollSeconds,
    [string]$Name
  )

  $previousLength = -1
  while ($true) {
    if (-not (Test-Path $Path)) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] Waiting for $Name`: $Path"
      Start-Sleep -Seconds $PollSeconds
      continue
    }

    $currentLength = Get-FileLengthOrMinusOne -Path $Path
    if ($currentLength -le 0) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] $Name exists but is empty/incomplete: size=$currentLength"
      Start-Sleep -Seconds $PollSeconds
      $previousLength = $currentLength
      continue
    }

    if ($currentLength -eq $previousLength) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] $Name is stable: $Path size=$currentLength"
      return
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Name found, waiting for stable size: size=$currentLength"
    $previousLength = $currentLength
    Start-Sleep -Seconds $PollSeconds
  }
}

if ((Test-Path $GenerationMetrics) -and (-not $Overwrite)) {
  Write-Host "Latent generation metrics already exist, skipping: $GenerationMetrics"
  Write-Host "Use -Overwrite to rerun latent generation evaluation."
  exit 0
}

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $TestCache)) {
  throw "Test cache not found: $TestCache"
}
if (-not (Test-Path $Checkpoint)) {
  throw "Checkpoint not found: $Checkpoint"
}

Wait-ForStableFile -Path $EvalMetrics -PollSeconds $PollSeconds -Name "test eval metrics"
Wait-ForStableFile -Path $Checkpoint -PollSeconds $PollSeconds -Name "best checkpoint"

if ((Test-Path $GenerationMetrics) -and $Overwrite) {
  Write-Host "Overwrite requested; existing generation metrics will be replaced: $GenerationMetrics"
}

Write-Host "Starting SST-5 latent generation evaluation."
Write-Host "  model:      $ModelPath"
Write-Host "  checkpoint: $Checkpoint"
Write-Host "  test cache: $TestCache"
Write-Host "  output:     $GenerationOutputDir"

$evalArgs = @(
  "src/eval_sst5_latent_generation.py",
  "--model-path", $ModelPath,
  "--checkpoint", $Checkpoint,
  "--test-cache", $TestCache,
  "--output-dir", $GenerationOutputDir,
  "--max-new-tokens", $MaxNewTokens,
  "--temperature", $Temperature,
  "--device", $Device,
  "--load-dtype", $LoadDtype
)

if ($TrustRemoteCode) {
  $evalArgs += "--trust-remote-code"
}
if ($RunTeacherKBaseline) {
  $evalArgs += "--run-teacher-k-baseline"
}

& $PythonExe @evalArgs

if ($LASTEXITCODE -ne 0) {
  throw "Latent generation evaluation failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $GenerationMetrics)) {
  throw "Latent generation finished but metrics were not written: $GenerationMetrics"
}

Write-Host "Latent generation evaluation complete: $GenerationMetrics"
