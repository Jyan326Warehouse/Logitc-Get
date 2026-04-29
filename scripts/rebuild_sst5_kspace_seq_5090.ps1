param(
  [string]$PythonExe = "D:\Junyi_Files\conda-envs\logitc-get\python.exe",
  [string]$ModelPath = "models/Qwen3-8B",
  [string]$DataDir = "data/sst5",
  [string]$CacheDir = "data/sst5/teacher_logits/Qwen3-8B_kspace_seq_5090",
  [string]$OutputDir = "outputs/sst5_kspace_seq_qwen3_5090_lr5e-5",

  [int]$ExportBatchSize = 8,
  [int]$TeacherForcedBatchSize = 8,
  [int]$TrainBatchSize = 16,
  [int]$EvalBatchSize = 16,
  [int]$Epochs = 20,
  [double]$Lr = 0.00005,
  [double]$WeightDecay = 0.0001,
  [int]$FusionHiddenDim = 512,
  [int]$EncoderHiddenDim = 512,
  [int]$DecoderHiddenDim = 512,
  [double]$Dropout = 0.1,
  [int]$CotMaxNewTokens = 96,
  [int]$LatentMaxNewTokens = 64,
  [string]$Device = "cuda",
  [ValidateSet("float16", "bfloat16", "float32")]
  [string]$TeacherLoadDtype = "bfloat16",
  [ValidateSet("float16", "bfloat16", "float32")]
  [string]$CacheSaveDtype = "float16",
  [ValidateSet("float16", "bfloat16", "float32")]
  [string]$TrainLogitsDtype = "float16",
  [switch]$TrustRemoteCode,
  [switch]$OverwriteCache,
  [switch]$SkipExport,
  [switch]$SkipTrain,
  [switch]$SkipTestEval,
  [switch]$SkipLatentGeneration
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  $start = Get-Date
  Write-Host ""
  Write-Host "===== $Name | started $($start.ToString('yyyy-MM-dd HH:mm:ss')) ====="
  & $Body
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
  $end = Get-Date
  $elapsed = $end - $start
  Write-Host "===== $Name | finished $($end.ToString('yyyy-MM-dd HH:mm:ss')) | elapsed $elapsed ====="
}

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SkipExport) {
  $exportArgs = @(
    "src/export_sst5_teacher_logits.py",
    "--model-path", $ModelPath,
    "--data-dir", $DataDir,
    "--output-dir", $CacheDir,
    "--batch-size", $ExportBatchSize,
    "--prefix-batch-size", $TeacherForcedBatchSize,
    "--cot-max-new-tokens", $CotMaxNewTokens,
    "--load-dtype", $TeacherLoadDtype,
    "--save-dtype", $CacheSaveDtype
  )
  if ($TrustRemoteCode) { $exportArgs += "--trust-remote-code" }
  if ($OverwriteCache) { $exportArgs += "--overwrite" }
  Invoke-Step "export frozen K-space teacher cache" {
    & $PythonExe @exportArgs
  }
}

if (-not $SkipTrain) {
  Invoke-Step "train SST-5 K-space AE" {
    & $PythonExe "src/train_sst5_content_ae.py" `
      --train-cache (Join-Path $CacheDir "train.pt") `
      --val-cache (Join-Path $CacheDir "validation.pt") `
      --output-dir $OutputDir `
      --batch-size $TrainBatchSize `
      --epochs $Epochs `
      --lr $Lr `
      --weight-decay $WeightDecay `
      --fusion-hidden-dim $FusionHiddenDim `
      --encoder-hidden-dim $EncoderHiddenDim `
      --decoder-hidden-dim $DecoderHiddenDim `
      --dropout $Dropout `
      --ce-weight 0.5 `
      --kl-weight 0.5 `
      --device $Device `
      --logits-dtype $TrainLogitsDtype
  }
}

if (-not $SkipTestEval) {
  Invoke-Step "teacher-forced test eval" {
    & $PythonExe "src/eval_sst5_content_ae.py" `
      --cache (Join-Path $CacheDir "test.pt") `
      --checkpoint (Join-Path $OutputDir "best_sst5_content_ae.pt") `
      --output-dir (Join-Path $OutputDir "test_eval") `
      --batch-size $EvalBatchSize `
      --device $Device `
      --logits-dtype $TrainLogitsDtype `
      --split-name "test"
  }
}

if (-not $SkipLatentGeneration) {
  $latentArgs = @(
    "src/eval_sst5_latent_generation.py",
    "--model-path", $ModelPath,
    "--checkpoint", (Join-Path $OutputDir "best_sst5_content_ae.pt"),
    "--test-cache", (Join-Path $CacheDir "test.pt"),
    "--output-dir", (Join-Path $OutputDir "latent_generation_eval"),
    "--max-new-tokens", $LatentMaxNewTokens,
    "--temperature", "1.0",
    "--device", $Device,
    "--load-dtype", $TeacherLoadDtype
  )
  if ($TrustRemoteCode) { $latentArgs += "--trust-remote-code" }
  Invoke-Step "full latent generation eval" {
    & $PythonExe @latentArgs
  }
}

Write-Host ""
Write-Host "Rebuild complete."
Write-Host "CacheDir:  $CacheDir"
Write-Host "OutputDir: $OutputDir"
