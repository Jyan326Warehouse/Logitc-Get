# GSM8K Logits AutoEncoder

This project exports token-level teacher logits from GSM8K and trains a logits
autoencoder on those vectors.

## Overview

The current default setup is a lightweight experiment:

- source dataset: `openai/gsm8k`, config `main`
- train source samples: `3200`
- test source samples: `800`
- input dimension: `151936`
- latent dimension: `256`
- loss: MSE only

The training pipeline is:

1. Export teacher logits from GSM8K examples.
2. Convert each sample's `answer_logits[T, V]` into token-level AE samples.
3. Train the logits autoencoder on train tokens.
4. Evaluate the trained AE on test tokens.

## Repository Layout

- `src/export_teacher_logits.py`: export token-level teacher logits and metadata
- `src/dataset.py`: build token-level training samples from exported logits
- `src/model.py`: define the logits autoencoder
- `src/train.py`: train the AE on GPU
- `src/eval.py`: evaluate a saved checkpoint
- `data/meta/`: lightweight metadata index files
- `data/logits/`: generated logits files, ignored by git
- `outputs/`: checkpoints, ignored by git

## Data Export

Export the default lightweight train split:

```bash
python3 src/export_teacher_logits.py \
  --model_path /path/to/Qwen3-8B \
  --split train \
  --output_root data \
  --load_dtype bfloat16 \
  --save_dtype float16 \
  --trust_remote_code
```

Export the default lightweight test split:

```bash
python3 src/export_teacher_logits.py \
  --model_path /path/to/Qwen3-8B \
  --split test \
  --output_root data \
  --load_dtype bfloat16 \
  --save_dtype float16 \
  --trust_remote_code
```

To export the full split instead of the lightweight cap, add `--full_split`.

## Training

Run training on the default lightweight configuration:

```bash
python3 src/train.py --split train --test-split test
```

Run training in the `COT_CNN` conda environment with `nohup`:

```bash
nohup bash -lc 'source /home/yangtong/miniconda3/etc/profile.d/conda.sh && conda activate COT_CNN && python -u src/train.py' \
  > train.log 2>&1 &
```

Training logs now include:

- interval MSE
- average step time
- samples per second
- elapsed time
- ETA
- per-epoch training time
- evaluation time

## Evaluation

Evaluate the latest checkpoint on the default test subset:

```bash
python3 src/eval.py --checkpoint outputs/latest.pt
```

## Notes

Large generated files are intentionally ignored by git:

- `data/logits/`
- `outputs/`
- `*.pt`
- `*.log`
