# GSM8K Logits AutoEncoder

This project exports token-level teacher logits for GSM8K and trains a logits
autoencoder on those vectors.

Default lightweight experiment setup:

- Export teacher logits from `train` and `test`.
- Use 4000 GSM8K source samples total by default.
- Use `3200` source samples from `train`.
- Use `800` source samples from `test`.
- Train the AE on the capped `train` subset.
- Evaluate the AE on the capped `test` subset.
- Use input dimension `151936` and latent dimension `256`.
- Use MSE reconstruction loss only.

Large generated files are intentionally ignored by git:

- `data/logits/`
- `outputs/`
- `*.pt`
- `*.log`

Example training command:

```bash
python3 src/train.py --split train --test-split test
```

Export the default lightweight logits:

```bash
python3 src/export_teacher_logits.py --model_path /path/to/Qwen3-8B --split train --trust_remote_code
python3 src/export_teacher_logits.py --model_path /path/to/Qwen3-8B --split test --trust_remote_code
```

Pass `--full_split` to export the complete split instead of the default lightweight cap.
