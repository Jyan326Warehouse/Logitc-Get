# Logitc-Get

Active branch: **GSM Token-List AE**

The project mainline is now:

```text
teacher full logits [T, V]
-> GSM8K-specific token list T_GSM
-> projected token-list logits [T, K]
-> token-aligned latent AE with latent dim K
-> KL + latent CE + top-k evaluation
```

The old full-vocabulary AE path and hidden-state AE path are no longer active
and have been removed from the clean source tree. Existing exported `.pt`
teacher logits are treated as generated artifacts and are not committed.

## Active Files

- `src/export_teacher_logits.py`: export teacher `answer_logits: [T, V]` and answer token ids.
- `src/build_gsm_token_list.py`: build a data-driven all-text GSM token list.
- `src/gsm_tokenlist_dataset.py`: project full logits into K-dimensional token-list space.
- `src/tokenlist_ae_model.py`: define `GSMTokenListAE`, with latent dim exactly `K`.
- `src/train_tokenlist_ae.py`: train KL + latent CE token-list AE.
- `src/eval_tokenlist_ae.py`: evaluate KL, CE, top-k accuracy, rank, and OOV stats.
- `README_tokenlist_ae.md`: full command flow.

## Quick Flow

```powershell
python src/export_teacher_logits.py `
  --model-path models/Qwen3-8B `
  --input-meta data/meta/ae_train.jsonl `
  --output-split ae_train `
  --output_root data `
  --trust_remote_code

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

python src/eval_tokenlist_ae.py `
  --logits-dir data/logits/main_test `
  --token-list-json outputs/gsm_token_list_all_text/gsm_token_list.json `
  --checkpoint outputs/tokenlist_ae_all_text/best_tokenlist_ae.pt `
  --output-dir outputs/tokenlist_ae_all_text/test_eval `
  --split-name test
```

Expected primary outputs:

```text
outputs/gsm_token_list_all_text/gsm_token_list.json
outputs/gsm_token_list_all_text/token_list_coverage_report.json
outputs/tokenlist_ae_all_text/best_tokenlist_ae.pt
outputs/tokenlist_ae_all_text/train_metrics.jsonl
outputs/tokenlist_ae_all_text/final_metrics.json
outputs/tokenlist_ae_all_text/test_eval/eval_metrics.json
outputs/tokenlist_ae_all_text/test_eval/eval_predictions_sample.jsonl
```

See `README_tokenlist_ae.md` for details.
