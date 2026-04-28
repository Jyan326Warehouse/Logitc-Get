param(
  [string]$PythonExe = "D:\Junyi_Files\conda-envs\logitc-get\python.exe",
  [string]$CacheDir = "data/sst5/teacher_logits/Qwen3-8B_kspace_seq",
  [string]$OutputDir = "outputs/sst5_kspace_seq_qwen3_lr5e-5",
  [int]$BatchSize = 8,
  [string]$Device = "cuda",
  [ValidateSet("float16", "bfloat16", "float32")]
  [string]$LogitsDtype = "float16",
  [int]$PollSeconds = 30,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

if ($PollSeconds -le 0) {
  throw "PollSeconds must be positive; got $PollSeconds"
}

$FinalCheckpoint = Join-Path $OutputDir "final_sst5_content_ae.pt"
$BestCheckpoint = Join-Path $OutputDir "best_sst5_content_ae.pt"
$EvalOutputDir = Join-Path $OutputDir "test_eval"
$EvalMetrics = Join-Path $EvalOutputDir "eval_metrics.json"
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
    [int]$PollSeconds
  )

  $previousLength = -1
  while ($true) {
    if (-not (Test-Path $Path)) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] Waiting for final checkpoint: $Path"
      Start-Sleep -Seconds $PollSeconds
      continue
    }

    $currentLength = Get-FileLengthOrMinusOne -Path $Path
    if ($currentLength -le 0) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] Final checkpoint exists but is empty/incomplete: size=$currentLength"
      Start-Sleep -Seconds $PollSeconds
      $previousLength = $currentLength
      continue
    }

    if ($currentLength -eq $previousLength) {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      Write-Host "[$timestamp] Final checkpoint is stable: $Path size=$currentLength"
      return
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Final checkpoint found, waiting for stable size: size=$currentLength"
    $previousLength = $currentLength
    Start-Sleep -Seconds $PollSeconds
  }
}

if ((Test-Path $EvalMetrics) -and (-not $Overwrite)) {
  Write-Host "Eval metrics already exist, skipping: $EvalMetrics"
  Write-Host "Use -Overwrite to rerun test evaluation."
  exit 0
}

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $TestCache)) {
  throw "Test cache not found: $TestCache"
}

Wait-ForStableFile -Path $FinalCheckpoint -PollSeconds $PollSeconds

if (-not (Test-Path $BestCheckpoint)) {
  throw "Best checkpoint not found after train completion: $BestCheckpoint"
}
Wait-ForStableFile -Path $BestCheckpoint -PollSeconds $PollSeconds

if ((Test-Path $EvalMetrics) -and $Overwrite) {
  Write-Host "Overwrite requested; existing eval metrics will be replaced: $EvalMetrics"
}

Write-Host "Starting SST-5 test evaluation."
Write-Host "  cache:      $TestCache"
Write-Host "  checkpoint: $BestCheckpoint"
Write-Host "  output:     $EvalOutputDir"

& $PythonExe "src/eval_sst5_content_ae.py" `
  --cache $TestCache `
  --checkpoint $BestCheckpoint `
  --output-dir $EvalOutputDir `
  --batch-size $BatchSize `
  --device $Device `
  --logits-dtype $LogitsDtype `
  --split-name "test"

if ($LASTEXITCODE -ne 0) {
  throw "Test evaluation failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $EvalMetrics)) {
  throw "Test evaluation finished but metrics were not written: $EvalMetrics"
}

Write-Host "Test evaluation complete: $EvalMetrics"
