# GSM Token-List AE

This is the active experiment path for AE/logit compression.

The project no longer trains a full-vocabulary autoencoder such as:

```text
full logits V -> latent 256 -> full logits V
```

The active path is:

```text
teacher full logits [T, V]
-> build GSM8K-specific token list T_GSM from full train text
-> gather/project logits into token-list space [T, K]
-> train token-aligned latent AE where latent dim = K
-> evaluate KL + latent CE + top-k accuracy
```

Every latent dimension is aligned to the same token id in `gsm_token_list.json`.
`K` is data-driven: it equals the number of unique token ids from the full
GSM8K train text plus any train answer targets that must be forced in for CE.

## Step 1: Export Teacher Logits

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Use the existing teacher logits exporter. It writes `answer_logits: [T, V]` and both
`answer_token_ids: [T]` and legacy-compatible `answer_ids: [T]`.

```powershell
python src/export_teacher_logits.py `
  --model-path models/Qwen3-8B `
  --input-meta data/meta/ae_train.jsonl `
  --output-split ae_train `
  --output_root data `
  --trust_remote_code
```

Repeat with `data/meta/ae_val.jsonl -> --output-split ae_val` and
`data/meta/main_test.jsonl -> --output-split main_test`.

The new token-list scripts also accept older exported `.pt` files that only contain
`answer_ids`.

## Step 2: Build GSM Token List From Full Train Text

```powershell
python src/build_gsm_token_list.py `
  --build-mode all_text `
  --train-meta-jsonl data/meta/ae_train.jsonl `
  --val-meta-jsonl data/meta/ae_val.jsonl `
  --test-meta-jsonl data/meta/main_test.jsonl `
  --train-logits-dir data/logits/ae_train `
  --val-logits-dir data/logits/ae_val `
  --test-logits-dir data/logits/main_test `
  --tokenizer-path models/Qwen3-8B `
  --output-dir outputs/gsm_token_list_all_text
```

Outputs:

```text
outputs/gsm_token_list_all_text/gsm_token_list.json
outputs/gsm_token_list_all_text/token_list_coverage_report.json
```

Token-list construction:

1. Read `--train-meta-jsonl`.
2. Build full text for each train sample, preferring fields such as `question`,
   `answer`, `rationale`, `final_answer`, `prompt`, `messages`, `text`,
   `input`, and `output`. Existing `prompt_text + answer_text` records are
   treated as the actual training/inference format.
3. Tokenize full train text with `--tokenizer-path`.
4. Set `T_GSM = unique(tokenize(full train text))`.
5. Scan `--train-logits-dir` answer targets and force-add any missing
   `answer_token_ids`.
6. Sort by text token frequency descending, then token id ascending.

The token list is not truncated and does not use teacher top-k. If
`--train-meta-jsonl` is omitted, the script falls back to answer-token-only
construction and prints a warning.

## Step 3: Train Token-List AE

```powershell
python src/train_tokenlist_ae.py `
  --train-logits-dir data/logits/ae_train `
  --val-logits-dir data/logits/ae_val `
  --token-list-json outputs/gsm_token_list_all_text/gsm_token_list.json `
  --output-dir outputs/tokenlist_ae_all_text `
  --batch-size 64 `
  --epochs 20 `
  --lr 1e-4 `
  --hidden-dim 1024 `
  --lambda-kl 1.0 `
  --lambda-latent-ce 1.0 `
  --lambda-recon-ce 0.0
```

Main losses:

```text
KL:        KL(softmax(input_logits_k / T) || softmax(recon_logits_k / T))
latent CE: cross_entropy(latent_logits_k, target_idx)
recon CE:  cross_entropy(recon_logits_k, target_idx), optional
```

Outputs:

```text
outputs/tokenlist_ae_all_text/best_tokenlist_ae.pt
outputs/tokenlist_ae_all_text/train_metrics.jsonl
outputs/tokenlist_ae_all_text/final_metrics.json
```

## Step 4: Evaluate

```powershell
python src/eval_tokenlist_ae.py `
  --logits-dir data/logits/main_test `
  --token-list-json outputs/gsm_token_list_all_text/gsm_token_list.json `
  --checkpoint outputs/tokenlist_ae_all_text/best_tokenlist_ae.pt `
  --output-dir outputs/tokenlist_ae_all_text/test_eval `
  --split-name test
```

Outputs:

```text
outputs/tokenlist_ae_all_text/test_eval/eval_metrics.json
outputs/tokenlist_ae_all_text/test_eval/eval_predictions_sample.jsonl
```

## Notes

- Full vocabulary logits are read only as an upstream teacher source.
- Training batches contain only `input_logits_k: [B, K]`, never `[B, V]`.
- `latent_logits` and `recon_logits` are both K-dimensional and aligned to the same token list.
- OOV true tokens are skipped by default and reported in dataset/eval metrics.
- `token_list_coverage_report.json` reports full-text token coverage and
  answer-target coverage for train/val/test when the corresponding meta/logits
  inputs are provided.

## Generated Artifacts

Teacher logits, model weights, checkpoints, logs, and other large generated
artifacts are intentionally ignored by git. Regenerate them with the scripts in
`scripts/` when moving to a new machine.

## Reproduce Current 4000-Sample Run

The committed metadata split has 3000 train, 200 val, and 800 test samples. The
current run used `models/Qwen3-8B` as teacher/tokenizer and produced `K = 9839`
for the all-text GSM token list.

Run the full train pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_tokenlist_ae_train_pipeline.ps1 `
  -Python "D:\Junyi_Files\conda-envs\logitc-get\python.exe" `
  -ModelPath "models\Qwen3-8B"
```

Then run test evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_tokenlist_ae_test_eval.ps1 `
  -Python "D:\Junyi_Files\conda-envs\logitc-get\python.exe" `
  -ModelPath "models\Qwen3-8B"
```

If your conda environment is already activated, you can omit `-Python`; the
scripts default to `python`.

Expected files:

```text
outputs/gsm_token_list_all_text_rebuilt_4000/gsm_token_list.json
outputs/gsm_token_list_all_text_rebuilt_4000/token_list_coverage_report.json
outputs/tokenlist_ae_all_text_rebuilt_4000/best_tokenlist_ae.pt
outputs/tokenlist_ae_all_text_rebuilt_4000/train_metrics.jsonl
outputs/tokenlist_ae_all_text_rebuilt_4000/final_metrics.json
outputs/tokenlist_ae_all_text_rebuilt_4000/test_eval/eval_metrics.json
outputs/tokenlist_ae_all_text_rebuilt_4000/test_eval/eval_predictions_sample.jsonl
```
