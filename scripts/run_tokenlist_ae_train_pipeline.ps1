param(
  [string]$Python = "python",
  [string]$ModelPath = "models\Qwen3-8B"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$TokenListDir = "outputs\gsm_token_list_all_text_rebuilt_4000"
$TrainOutputDir = "outputs\tokenlist_ae_all_text_rebuilt_4000"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\logits" | Out-Null

$env:PYTHONUNBUFFERED = "1"

Write-Host "==== Token-list AE pipeline started: $(Get-Date -Format o) ===="
Write-Host "ProjectRoot=$ProjectRoot"
Write-Host "Python=$Python"
Write-Host "ModelPath=$ModelPath"

Write-Host "==== Step 1/4: export ae_train teacher logits ===="
& $Python "src\export_teacher_logits.py" `
  --model-path $ModelPath `
  --input-meta "data\meta\ae_train.jsonl" `
  --output-split "ae_train" `
  --output_root "data" `
  --load-dtype "bfloat16" `
  --save-dtype "float16" `
  --trust-remote-code `
  --skip-existing

Write-Host "==== Step 2/4: export ae_val teacher logits ===="
& $Python "src\export_teacher_logits.py" `
  --model-path $ModelPath `
  --input-meta "data\meta\ae_val.jsonl" `
  --output-split "ae_val" `
  --output_root "data" `
  --load-dtype "bfloat16" `
  --save-dtype "float16" `
  --trust-remote-code `
  --skip-existing

Write-Host "==== Step 3/4: rebuild final all-text token list with answer targets ===="
& $Python "src\build_gsm_token_list.py" `
  --build-mode "all_text" `
  --train-meta-jsonl "data\meta\ae_train.jsonl" `
  --val-meta-jsonl "data\meta\ae_val.jsonl" `
  --test-meta-jsonl "data\meta\main_test.jsonl" `
  --train-logits-dir "data\logits\ae_train" `
  --val-logits-dir "data\logits\ae_val" `
  --test-logits-dir "data\logits\main_test" `
  --tokenizer-path $ModelPath `
  --output-dir $TokenListDir `
  --log-every 100

Write-Host "==== Step 4/4: train token-list AE ===="
& $Python "src\train_tokenlist_ae.py" `
  --train-logits-dir "data\logits\ae_train" `
  --val-logits-dir "data\logits\ae_val" `
  --token-list-json "$TokenListDir\gsm_token_list.json" `
  --output-dir $TrainOutputDir `
  --batch-size 64 `
  --epochs 20 `
  --lr 1e-4 `
  --weight-decay 1e-5 `
  --hidden-dim 1024 `
  --dropout 0.1 `
  --temperature 1.0 `
  --lambda-kl 1.0 `
  --lambda-latent-ce 1.0 `
  --lambda-recon-ce 0.0 `
  --num-workers 0 `
  --device "auto" `
  --seed 42 `
  --preload `
  --logits-dtype "float16" `
  --log-every 100

Write-Host "==== Token-list AE pipeline finished: $(Get-Date -Format o) ===="
