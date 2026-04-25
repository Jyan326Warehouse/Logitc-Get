import argparse
import sys
import time
from math import ceil
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from dataset import TokenLogitsDataset
from model import build_model


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TRAIN_SAMPLES = 3200
DEFAULT_TEST_SAMPLES = 800
DEFAULT_CACHE_SIZE = 64
DEFAULT_NUM_WORKERS = 2


class RecordBatchSampler(Sampler):
    """
    Shuffle at the source-sample level while iterating token indices contiguously
    within each record. This avoids repeatedly reloading the same logits file for
    randomly scattered token indices.
    """

    def __init__(self, dataset, batch_size: int, shuffle_records: bool, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_records = shuffle_records
        self.seed = seed
        self.epoch = 0
        self.record_starts = [0]
        self.record_lengths = []

        prev = 0
        for cum in dataset.cum_token_counts:
            self.record_lengths.append(cum - prev)
            self.record_starts.append(cum)
            prev = cum
        self.record_starts = self.record_starts[:-1]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return sum(ceil(length / self.batch_size) for length in self.record_lengths)

    def __iter__(self):
        if self.shuffle_records:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            record_order = torch.randperm(len(self.record_starts), generator=generator).tolist()
        else:
            record_order = range(len(self.record_starts))

        for record_idx in record_order:
            start = self.record_starts[record_idx]
            length = self.record_lengths[record_idx]
            end = start + length
            for batch_start in range(start, end, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, end)))


def parse_args():
    parser = argparse.ArgumentParser(description="Train logits AE on token-level teacher logits.")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-config", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--test-max-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=151936)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle_records: bool, seed: int):
    batch_sampler = RecordBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle_records=shuffle_records,
        seed=seed,
    )
    kwargs = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def build_datasets(args):
    train_dataset = TokenLogitsDataset(
        data_root=args.data_root,
        dataset_config=args.dataset_config,
        split=args.split,
        logits_dtype=torch.float32,
        cache_size=args.cache_size,
        max_samples=args.max_samples,
    )

    test_split = args.test_split if args.test_split is not None else args.split
    test_max_samples = args.test_max_samples if args.test_max_samples is not None else args.max_samples

    if test_split == args.split and test_max_samples == args.max_samples:
        test_dataset = train_dataset
    else:
        test_dataset = TokenLogitsDataset(
            data_root=args.data_root,
            dataset_config=args.dataset_config,
            split=test_split,
            logits_dtype=torch.float32,
            cache_size=args.cache_size,
            max_samples=test_max_samples,
        )

    return train_dataset, test_dataset


def extract_inputs(batch, device: torch.device) -> torch.Tensor:
    return batch["x"].to(device, non_blocking=True)


def mse_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon_x, x, reduction="mean")


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def model_stats(model) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def run_epoch(model, loader, optimizer, device, epoch: int, log_interval: int):
    model.train()
    total_loss = 0.0
    total_samples = 0
    epoch_start = time.perf_counter()
    interval_start = epoch_start
    interval_loss = 0.0
    interval_samples = 0
    interval_steps = 0

    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(loader):
        x = extract_inputs(batch, device)
        optimizer.zero_grad()
        recon_x = model(x)
        loss = mse_loss(recon_x, x)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_samples = len(x)
        total_loss += batch_loss
        total_samples += batch_samples
        interval_loss += batch_loss
        interval_samples += batch_samples
        interval_steps += 1
        if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
            now = time.perf_counter()
            interval_time = max(now - interval_start, 1e-6)
            avg_step_time = interval_time / max(interval_steps, 1)
            samples_per_sec = interval_samples / interval_time
            processed_ratio = (batch_idx + 1) / len(loader)
            elapsed = now - epoch_start
            eta = (elapsed / processed_ratio) - elapsed if processed_ratio > 0 else 0.0
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tAvgStep: {:.3f}s\tSamples/s: {:.1f}\tElapsed: {}\tETA: {}".format(
                    epoch,
                    total_samples,
                    len(loader.dataset),
                    100.0 * (batch_idx + 1) / len(loader),
                    interval_loss / max(interval_steps, 1),
                    avg_step_time,
                    samples_per_sec,
                    format_duration(elapsed),
                    format_duration(eta),
                )
            )
            interval_start = now
            interval_loss = 0.0
            interval_samples = 0
            interval_steps = 0

    avg_loss = total_loss / len(loader)
    epoch_time = time.perf_counter() - epoch_start
    print(
        "====> Epoch: {} Average MSE: {:.6f}\tEpochTime: {}\tAvgSamples/s: {:.1f}".format(
            epoch,
            avg_loss,
            format_duration(epoch_time),
            total_samples / max(epoch_time, 1e-6),
        )
    )
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    eval_start = time.perf_counter()
    for batch in loader:
        x = extract_inputs(batch, device)
        recon_x = model(x)
        total_loss += mse_loss(recon_x, x).item()
        total_samples += len(x)

    avg_loss = total_loss / len(loader)
    eval_time = time.perf_counter() - eval_start
    print(
        "====> Test set MSE: {:.6f}\tEvalTime: {}\tEvalSamples/s: {:.1f}".format(
            avg_loss,
            format_duration(eval_time),
            total_samples / max(eval_time, 1e-6),
        )
    )
    return avg_loss


def save_checkpoint(model, optimizer, args, epoch: int, train_loss: float, test_loss: float):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"logits_ae_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "input_dim": args.input_dim,
            "latent_dim": args.latent_dim,
            "train_split": args.split,
            "test_split": args.test_split,
            "max_samples": args.max_samples,
            "test_max_samples": args.test_max_samples,
        },
        ckpt_path,
    )
    latest_path = output_dir / "latest.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "input_dim": args.input_dim,
            "latent_dim": args.latent_dim,
            "train_split": args.split,
            "test_split": args.test_split,
            "max_samples": args.max_samples,
            "test_max_samples": args.test_max_samples,
        },
        latest_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for logits AE training.")

    seed_everything(args.seed)
    device = torch.device("cuda")

    train_dataset, test_dataset = build_datasets(args)

    print(f"Train split: {args.split}")
    print(f"Test split: {args.test_split}")
    print(f"Train source sample cap: {args.max_samples}")
    print(f"Test source sample cap: {args.test_max_samples}")
    print(f"Loaded {len(train_dataset)} train token-level samples")
    print(f"Loaded {len(test_dataset)} test token-level samples")
    print("Batching mode: record-contiguous token batches for faster logits file reuse")

    model = build_model(input_dim=args.input_dim, latent_dim=args.latent_dim).to(device)
    total_params, trainable_params = model_stats(model)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = build_dataloader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle_records=True,
        seed=args.seed,
    )
    test_loader = build_dataloader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle_records=False,
        seed=args.seed,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)
        test_loss = evaluate(model, test_loader, device)
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, args, epoch, train_loss, test_loss)


if __name__ == "__main__":
    main()
