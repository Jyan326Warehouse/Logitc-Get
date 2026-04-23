import argparse
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import TokenLogitsDataset
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train logits AE on token-level teacher logits.")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-config", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--input-dim", type=int, default=151936)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
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


def run_epoch(model, loader, optimizer, device, epoch: int, log_interval: int):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        x = extract_inputs(batch, device)
        optimizer.zero_grad()
        recon_x = model(x)
        loss = mse_loss(recon_x, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}".format(
                    epoch,
                    batch_idx * len(x),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )

    avg_loss = total_loss / len(loader)
    print("====> Epoch: {} Average MSE: {:.6f}".format(epoch, avg_loss))
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x = extract_inputs(batch, device)
        recon_x = model(x)
        total_loss += mse_loss(recon_x, x).item()

    avg_loss = total_loss / len(loader)
    print("====> Test set MSE: {:.6f}".format(avg_loss))
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
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    test_loader = build_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    print(f"Train split: {args.split}")
    print(f"Test split: {args.test_split}")
    print(f"Loaded {len(train_dataset)} train token-level samples")
    print(f"Loaded {len(test_dataset)} test token-level samples")

    model = build_model(input_dim=args.input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)
        test_loss = evaluate(model, test_loader, device)
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, args, epoch, train_loss, test_loss)


if __name__ == "__main__":
    main()
