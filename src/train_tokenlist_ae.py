import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gsm_tokenlist_dataset import GSMTokenListDataset, load_token_list
from tokenlist_ae_model import GSMTokenListAEConfig, GSMTokenListAE


LOGGER = logging.getLogger("train_tokenlist_ae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GSM token-list aligned AE.")
    parser.add_argument("--train-logits-dir", required=True)
    parser.add_argument("--val-logits-dir", required=True)
    parser.add_argument("--token-list-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-latent-ce", type=float, default=1.0)
    parser.add_argument("--lambda-recon-ce", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--logits-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="CPU dtype used when preloading projected K-dimensional logits.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_arg]


def compute_losses(
    input_logits_k: torch.Tensor,
    target_idx: torch.Tensor,
    latent_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    temperature: float,
    lambda_kl: float,
    lambda_latent_ce: float,
    lambda_recon_ce: float,
) -> Dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    p = F.softmax(input_logits_k / temperature, dim=-1)
    log_q = F.log_softmax(recon_logits / temperature, dim=-1)
    loss_kl = F.kl_div(log_q, p, reduction="batchmean") * (temperature * temperature)
    loss_latent_ce = F.cross_entropy(latent_logits, target_idx)
    loss_recon_ce = F.cross_entropy(recon_logits, target_idx)
    loss = (
        lambda_kl * loss_kl
        + lambda_latent_ce * loss_latent_ce
        + lambda_recon_ce * loss_recon_ce
    )
    return {
        "loss": loss,
        "kl": loss_kl,
        "latent_ce": loss_latent_ce,
        "recon_ce": loss_recon_ce,
    }


def topk_correct(logits: torch.Tensor, target_idx: torch.Tensor, k: int) -> int:
    k_eff = min(int(k), logits.shape[-1])
    pred = logits.topk(k_eff, dim=-1).indices
    return int(pred.eq(target_idx.view(-1, 1)).any(dim=-1).sum().item())


def empty_totals() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "kl": 0.0,
        "latent_ce": 0.0,
        "recon_ce": 0.0,
        "latent_acc_top1": 0.0,
        "latent_acc_top5": 0.0,
        "recon_acc_top1": 0.0,
        "recon_acc_top5": 0.0,
        "num_samples": 0.0,
    }


def finalize_totals(totals: Dict[str, float]) -> Dict[str, float]:
    n = max(int(totals["num_samples"]), 1)
    return {
        "loss": totals["loss"] / n,
        "kl": totals["kl"] / n,
        "latent_ce": totals["latent_ce"] / n,
        "recon_ce": totals["recon_ce"] / n,
        "latent_acc_top1": totals["latent_acc_top1"] / n,
        "latent_acc_top5": totals["latent_acc_top5"] / n,
        "recon_acc_top1": totals["recon_acc_top1"] / n,
        "recon_acc_top5": totals["recon_acc_top5"] / n,
        "num_samples": int(totals["num_samples"]),
    }


def run_epoch(
    model: GSMTokenListAE,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    split_name: str = "train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = empty_totals()

    for step, batch in enumerate(loader, start=1):
        input_logits_k = batch["input_logits_k"].to(device, non_blocking=True).float()
        target_idx = batch["target_idx"].to(device, non_blocking=True).long()
        batch_size = int(target_idx.numel())

        with torch.set_grad_enabled(is_train):
            outputs = model(input_logits_k)
            losses = compute_losses(
                input_logits_k=input_logits_k,
                target_idx=target_idx,
                latent_logits=outputs["latent_logits"],
                recon_logits=outputs["recon_logits"],
                temperature=args.temperature,
                lambda_kl=args.lambda_kl,
                lambda_latent_ce=args.lambda_latent_ce,
                lambda_recon_ce=args.lambda_recon_ce,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()

        totals["loss"] += float(losses["loss"].detach().item()) * batch_size
        totals["kl"] += float(losses["kl"].detach().item()) * batch_size
        totals["latent_ce"] += float(losses["latent_ce"].detach().item()) * batch_size
        totals["recon_ce"] += float(losses["recon_ce"].detach().item()) * batch_size
        totals["latent_acc_top1"] += topk_correct(outputs["latent_logits"].detach(), target_idx, 1)
        totals["latent_acc_top5"] += topk_correct(outputs["latent_logits"].detach(), target_idx, 5)
        totals["recon_acc_top1"] += topk_correct(outputs["recon_logits"].detach(), target_idx, 1)
        totals["recon_acc_top5"] += topk_correct(outputs["recon_logits"].detach(), target_idx, 5)
        totals["num_samples"] += batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            partial = finalize_totals(totals)
            LOGGER.info(
                "%s step=%d samples=%d loss=%.6f kl=%.6f latent_ce=%.6f recon_ce=%.6f",
                split_name,
                step,
                partial["num_samples"],
                partial["loss"],
                partial["kl"],
                partial["latent_ce"],
                partial["recon_ce"],
            )

    return finalize_totals(totals)


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "num_samples"} | {
        f"{prefix}_num_samples": metrics["num_samples"]
    }


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_loader(
    dataset: GSMTokenListDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = output_dir / "train_metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    token_list = load_token_list(args.token_list_json)
    k = len(token_list["token_ids"])
    device = resolve_device(args.device)
    LOGGER.info("Using device=%s | K=%d", device, k)

    train_dataset = GSMTokenListDataset(
        logits_dir=args.train_logits_dir,
        token_list_json=args.token_list_json,
        max_samples=args.max_train_samples,
        skip_oov=True,
        preload=args.preload,
        logits_dtype=resolve_dtype(args.logits_dtype),
    )
    val_dataset = GSMTokenListDataset(
        logits_dir=args.val_logits_dir,
        token_list_json=args.token_list_json,
        max_samples=args.max_val_samples,
        skip_oov=True,
        preload=args.preload,
        logits_dtype=resolve_dtype(args.logits_dtype),
    )
    if len(train_dataset) == 0:
        raise ValueError("Train dataset has zero evaluated positions.")
    if len(val_dataset) == 0:
        raise ValueError("Val dataset has zero evaluated positions.")

    LOGGER.info(
        "Train dataset: samples=%d total_positions=%d skipped_oov=%d oov_rate=%.6f",
        len(train_dataset),
        train_dataset.total_positions,
        train_dataset.skipped_oov_count,
        train_dataset.oov_rate,
    )
    LOGGER.info(
        "Val dataset: samples=%d total_positions=%d skipped_oov=%d oov_rate=%.6f",
        len(val_dataset),
        val_dataset.total_positions,
        val_dataset.skipped_oov_count,
        val_dataset.oov_rate,
    )

    model_config = GSMTokenListAEConfig(k=k, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = GSMTokenListAE(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        pin_memory=pin_memory,
    )

    run_config = {
        "train_logits_dir": args.train_logits_dir,
        "val_logits_dir": args.val_logits_dir,
        "token_list_json": args.token_list_json,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "lambda_kl": args.lambda_kl,
        "lambda_latent_ce": args.lambda_latent_ce,
        "lambda_recon_ce": args.lambda_recon_ce,
        "num_workers": args.num_workers,
        "device": str(device),
        "seed": args.seed,
        "k": k,
        "logits_dtype": args.logits_dtype,
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_tokenlist_ae.pt"
    last_record: Dict = {}

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, args.epochs)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            args=args,
            optimizer=optimizer,
            split_name="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                optimizer=None,
                split_name="val",
            )

        record = {
            "epoch": epoch,
            **flatten_metrics("train", train_metrics),
            **flatten_metrics("val", val_metrics),
        }
        append_jsonl(metrics_jsonl, record)
        last_record = record

        LOGGER.info(
            "epoch=%d train_loss=%.6f train_kl=%.6f train_latent_ce=%.6f "
            "val_loss=%.6f val_kl=%.6f val_latent_ce=%.6f val_latent_acc_top1=%.6f",
            epoch,
            record["train_loss"],
            record["train_kl"],
            record["train_latent_ce"],
            record["val_loss"],
            record["val_kl"],
            record["val_latent_ce"],
            record["val_latent_acc_top1"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": run_config,
                "model_config": model_config.to_dict(),
                "token_list_json": str(Path(args.token_list_json)),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }
            torch.save(checkpoint, best_checkpoint_path)
            LOGGER.info("Saved best checkpoint to %s", best_checkpoint_path)

    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "last_epoch": args.epochs,
        "last_metrics": last_record,
        "train_dataset": {
            "samples": len(train_dataset),
            "total_positions": train_dataset.total_positions,
            "skipped_oov_count": train_dataset.skipped_oov_count,
            "oov_rate": train_dataset.oov_rate,
        },
        "val_dataset": {
            "samples": len(val_dataset),
            "total_positions": val_dataset.total_positions,
            "skipped_oov_count": val_dataset.skipped_oov_count,
            "oov_rate": val_dataset.oov_rate,
        },
        "config": run_config,
        "checkpoint": str(best_checkpoint_path),
    }
    final_metrics_path = output_dir / "final_metrics.json"
    write_json(final_metrics_path, final_metrics)
    LOGGER.info("Wrote %s", final_metrics_path)


if __name__ == "__main__":
    main()
