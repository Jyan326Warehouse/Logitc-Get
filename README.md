# GSM8K Logits AutoEncoder

This project exports token-level teacher logits for GSM8K and trains a logits
autoencoder on those vectors.

Default experiment setup:

- Export teacher logits from `train` and `test`.
- Train the AE on `train`.
- Evaluate the AE on `test`.
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
