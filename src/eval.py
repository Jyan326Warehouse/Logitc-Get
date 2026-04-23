import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import TokenLogitsDataset
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate logits AE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-config", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for logits AE evaluation.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    input_dim = checkpoint.get("input_dim", 151936)
    latent_dim = checkpoint.get("latent_dim", 256)

    model = build_model(input_dim=input_dim, latent_dim=latent_dim).to("cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TokenLogitsDataset(
        data_root=args.data_root,
        dataset_config=args.dataset_config,
        split=args.split,
        logits_dtype=torch.float32,
        cache_size=args.cache_size,
        max_samples=args.max_samples,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to("cuda", non_blocking=True)
        recon_x = model(x)
        total_loss += F.mse_loss(recon_x, x, reduction="mean").item()

    avg_loss = total_loss / len(loader)
    print(f"Eval token-level samples: {len(dataset)}")
    print(f"MSE: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
