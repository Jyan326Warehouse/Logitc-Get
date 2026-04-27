$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = "D:\Junyi_Files\conda-envs\logitc-get\python.exe"
$ModelPath = "models\Qwen3-8B"
$TokenListJson = "outputs\gsm_token_list_all_text_rebuilt_4000\gsm_token_list.json"
$Checkpoint = "outputs\tokenlist_ae_all_text_rebuilt_4000\best_tokenlist_ae.pt"
$EvalOutputDir = "outputs\tokenlist_ae_all_text_rebuilt_4000\test_eval"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$env:PYTHONUNBUFFERED = "1"

Write-Host "==== Token-list AE test eval started: $(Get-Date -Format o) ===="

Write-Host "==== Step 1/2: export main_test teacher logits ===="
& $Python "src\export_teacher_logits.py" `
  --model-path $ModelPath `
  --input-meta "data\meta\main_test.jsonl" `
  --output-split "main_test" `
  --output_root "data" `
  --load-dtype "bfloat16" `
  --save-dtype "float16" `
  --trust-remote-code `
  --skip-existing

Write-Host "==== Step 2/2: evaluate best token-list AE checkpoint on test ===="
& $Python "src\eval_tokenlist_ae.py" `
  --logits-dir "data\logits\main_test" `
  --token-list-json $TokenListJson `
  --checkpoint $Checkpoint `
  --output-dir $EvalOutputDir `
  --split-name "test" `
  --batch-size 64 `
  --num-workers 0 `
  --device "auto" `
  --sample-size 100 `
  --preload

Write-Host "==== Token-list AE test eval finished: $(Get-Date -Format o) ===="
