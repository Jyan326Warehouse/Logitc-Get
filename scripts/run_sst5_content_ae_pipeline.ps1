param(
  [Parameter(Mandatory = $true)]
  [string]$ModelPath,

  [string]$PythonExe = "D:\Junyi_Files\conda-envs\logitc-get\python.exe",
  [string]$DataDir = "data/sst5",
  [string]$TeacherName = "",
  [string]$CacheDir = "",
  [string]$OutputDir = "outputs/sst5_content_ae",
  [int]$BatchSize = 8,
  [int]$PrefixBatchSize = 4,
  [int]$TrainBatchSize = 16,
  [int]$Epochs = 20,
  [int]$MinTokenCount = 1,
  [int]$MaxK = 0,
  [int]$CotMaxNewTokens = 96,
  [double]$Lr = 0.00005,
  [switch]$TrustRemoteCode,
  [switch]$OverwriteTeacherCache
)

$ErrorActionPreference = "Stop"

if ($TeacherName -eq "") {
  $TeacherName = Split-Path -Leaf $ModelPath
  $TeacherName = $TeacherName -replace "[^A-Za-z0-9._-]+", "_"
}
if ($CacheDir -eq "") {
  $CacheDir = Join-Path "data/sst5/teacher_logits" $TeacherName
}

$exportArgs = @(
  "src/export_sst5_teacher_logits.py",
  "--model-path", $ModelPath,
  "--data-dir", $DataDir,
  "--output-dir", $CacheDir,
  "--batch-size", $BatchSize,
  "--prefix-batch-size", $PrefixBatchSize,
  "--cot-max-new-tokens", $CotMaxNewTokens,
  "--min-token-count", $MinTokenCount
)
if ($MaxK -gt 0) { $exportArgs += @("--max-k", $MaxK) }
if ($TrustRemoteCode) { $exportArgs += "--trust-remote-code" }
if ($OverwriteTeacherCache) { $exportArgs += "--overwrite" }

& $PythonExe @exportArgs

& $PythonExe "src/train_sst5_content_ae.py" `
  --train-cache (Join-Path $CacheDir "train.pt") `
  --val-cache (Join-Path $CacheDir "validation.pt") `
  --output-dir $OutputDir `
  --batch-size $TrainBatchSize `
  --epochs $Epochs `
  --lr $Lr `
  --ce-weight 0.5 `
  --kl-weight 0.5

& $PythonExe "src/eval_sst5_content_ae.py" `
  --cache (Join-Path $CacheDir "test.pt") `
  --checkpoint (Join-Path $OutputDir "best_sst5_content_ae.pt") `
  --output-dir (Join-Path $OutputDir "test_eval") `
  --split-name "test"
