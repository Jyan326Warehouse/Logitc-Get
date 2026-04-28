from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sst5_common import project_path, read_json, write_json
from sst5_dataset import (
    SST5TeacherLogitsDataset,
    make_target_onehot_K,
    sst5_k_space_collate_fn,
)
from sst5_content_ae_model import SST5ContentAE, SST5ContentAEConfig


LOGGER = logging.getLogger("train_sst5_content_ae")


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=None)
    known, remaining = base.parse_known_args()
    defaults: Dict = {}
    if known.config is not None:
        defaults = read_json(known.config)

    def cfg(name: str, default):
        return defaults.get(name, default)

    parser = argparse.ArgumentParser(
        description="Train SST-5 AE fully in frozen token K-space: token CE + logitc KL.",
        parents=[base],
    )
    parser.add_argument("--train-cache", default=cfg("train_cache", None))
    parser.add_argument("--val-cache", default=cfg("val_cache", None))
    parser.add_argument("--output-dir", default=cfg("output_dir", "outputs/sst5_content_ae"))
    parser.add_argument("--expected-num-labels", type=int, default=cfg("expected_num_labels", 5))
    parser.add_argument("--batch-size", type=int, default=cfg("batch_size", 16))
    parser.add_argument("--epochs", type=int, default=cfg("epochs", 20))
    parser.add_argument("--lr", type=float, default=cfg("lr", 5e-5))
    parser.add_argument("--weight-decay", type=float, default=cfg("weight_decay", 1e-4))
    parser.add_argument("--fusion-hidden-dim", type=int, default=cfg("fusion_hidden_dim", 512))
    parser.add_argument("--encoder-hidden-dim", type=int, default=cfg("encoder_hidden_dim", 512))
    parser.add_argument("--decoder-hidden-dim", type=int, default=cfg("decoder_hidden_dim", 512))
    parser.add_argument("--dropout", type=float, default=cfg("dropout", 0.1))
    parser.add_argument("--no-residual-fusion", action="store_true", default=cfg("no_residual_fusion", False))
    parser.add_argument("--temperature", type=float, default=cfg("temperature", 1.0))
    parser.add_argument(
        "--ce-weight",
        "--lambda-ce",
        dest="ce_weight",
        type=float,
        default=cfg("ce_weight", cfg("lambda_ce", 0.5)),
    )
    parser.add_argument(
        "--kl-weight",
        "--lambda-kl",
        dest="kl_weight",
        type=float,
        default=cfg("kl_weight", cfg("lambda_kl", 0.5)),
    )
    parser.add_argument(
        "--verify-target-onehot",
        action=argparse.BooleanOptionalAction,
        default=cfg("verify_target_onehot", True),
        help="Build target_onehot_K inside the train step and validate [B,T,K] shape.",
    )
    parser.add_argument("--num-workers", type=int, default=cfg("num_workers", 0))
    parser.add_argument("--device", default=cfg("device", "auto"))
    parser.add_argument("--seed", type=int, default=cfg("seed", 42))
    parser.add_argument("--log-every", type=int, default=cfg("log_every", 50))
    parser.add_argument(
        "--logits-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=cfg("logits_dtype", "float32"),
    )
    args = parser.parse_args(remaining, namespace=known)
    for name in ("train_cache", "val_cache"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required")
    return args


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.shape]


def masked_mean(values: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"{name} values shape {_shape(values)} must match mask {_shape(mask)}")
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum()
    if float(denom.detach().cpu().item()) <= 0:
        raise ValueError(f"{name} has no valid tokens after applying loss_mask")
    return (values * mask_f).sum() / denom


def compute_k_space_loss(
    latent_logits_K: torch.Tensor,
    recon_logitc_K: torch.Tensor,
    input_logitc_K: torch.Tensor,
    target_k_indices: Optional[torch.Tensor] = None,
    target_onehot_K: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    ce_weight: float = 0.5,
    kl_weight: float = 0.5,
    temperature: float = 1.0,
    verify_target_onehot: bool = True,
) -> Dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if latent_logits_K.ndim != 3:
        raise ValueError(f"latent_logits_K must be [B,T,K], got {_shape(latent_logits_K)}")
    if recon_logitc_K.shape != latent_logits_K.shape:
        raise ValueError(
            "recon_logitc_K shape must equal latent_logits_K shape; "
            f"got recon={_shape(recon_logitc_K)} latent={_shape(latent_logits_K)}"
        )
    if input_logitc_K.shape != recon_logitc_K.shape:
        raise ValueError(
            "input_logitc_K shape must equal recon_logitc_K shape; "
            f"got input={_shape(input_logitc_K)} recon={_shape(recon_logitc_K)}"
        )
    bsz, steps, k = latent_logits_K.shape

    if loss_mask is None:
        if target_k_indices is None:
            raise ValueError("loss_mask is required when target_k_indices is not provided")
        loss_mask = target_k_indices.ne(-100)
    if loss_mask.shape != (bsz, steps):
        raise ValueError(
            f"loss_mask must be [B,T]=[{bsz},{steps}], got {_shape(loss_mask)}"
        )
    loss_mask = loss_mask.bool()

    if target_onehot_K is not None and target_onehot_K.shape != latent_logits_K.shape:
        raise ValueError(
            "target_onehot_K must be [B,T,K] matching latent_logits_K; "
            f"got target_onehot={_shape(target_onehot_K)} latent={_shape(latent_logits_K)}"
        )
    if target_k_indices is not None and target_k_indices.shape != (bsz, steps):
        raise ValueError(
            f"target_k_indices must be [B,T]=[{bsz},{steps}], got {_shape(target_k_indices)}"
        )
    if target_k_indices is None and target_onehot_K is None:
        raise ValueError("Either target_k_indices or target_onehot_K must be provided")

    if target_k_indices is not None:
        valid_ce_mask = loss_mask & target_k_indices.ne(-100)
        if bool(valid_ce_mask.any().item()):
            valid_targets = target_k_indices[valid_ce_mask]
            min_target = int(valid_targets.min().detach().cpu().item())
            max_target = int(valid_targets.max().detach().cpu().item())
            if min_target < 0 or max_target >= k:
                raise ValueError(
                    f"target_k_indices contains target outside K=[0,{k - 1}]: "
                    f"min={min_target} max={max_target}"
                )
        if verify_target_onehot and target_onehot_K is None:
            target_onehot_K = make_target_onehot_K(
                target_k_indices=target_k_indices,
                loss_mask=loss_mask,
                k=int(k),
                dtype=latent_logits_K.dtype,
            )
        ce_flat = F.cross_entropy(
            latent_logits_K.reshape(-1, k).float(),
            target_k_indices.reshape(-1).long(),
            ignore_index=-100,
            reduction="none",
        ).view(bsz, steps)
        loss_ce = masked_mean(ce_flat, valid_ce_mask, "K-space CE")
        token_acc_mask = valid_ce_mask
        pred_k = latent_logits_K.detach().argmax(dim=-1)
        token_correct = pred_k.eq(target_k_indices).logical_and(token_acc_mask).sum()
        token_total = token_acc_mask.sum()
    else:
        if target_onehot_K is None:
            raise ValueError("target_onehot_K is required when target_k_indices is absent")
        log_probs_K = F.log_softmax(latent_logits_K.float(), dim=-1)
        ce_per_token = -(target_onehot_K.float() * log_probs_K).sum(dim=-1)
        loss_ce = masked_mean(ce_per_token, loss_mask, "K-space CE")
        target_k = target_onehot_K.argmax(dim=-1)
        pred_k = latent_logits_K.detach().argmax(dim=-1)
        token_correct = pred_k.eq(target_k).logical_and(loss_mask).sum()
        token_total = loss_mask.sum()

    teacher_probs_K = F.softmax(input_logitc_K.float() / temperature, dim=-1)
    student_log_probs_K = F.log_softmax(recon_logitc_K.float() / temperature, dim=-1)
    loss_kl_per_token = (
        F.kl_div(student_log_probs_K, teacher_probs_K, reduction="none").sum(dim=-1)
        * (temperature**2)
    )
    loss_kl = masked_mean(loss_kl_per_token, loss_mask, "K-space KL")
    loss = float(ce_weight) * loss_ce + float(kl_weight) * loss_kl
    return {
        "loss": loss,
        "loss_ce": loss_ce,
        "loss_kl": loss_kl,
        "token_correct": token_correct.detach(),
        "token_total": token_total.detach(),
    }


def empty_totals() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "loss_ce": 0.0,
        "loss_kl": 0.0,
        "token_correct": 0.0,
        "token_total": 0.0,
        "num_samples": 0.0,
    }


def finalize_totals(totals: Dict[str, float]) -> Dict:
    token_total = max(float(totals["token_total"]), 1.0)
    return {
        "loss": totals["loss"] / token_total,
        "loss_ce": totals["loss_ce"] / token_total,
        "loss_kl": totals["loss_kl"] / token_total,
        "token_acc_K": totals["token_correct"] / token_total,
        "token_total": int(totals["token_total"]),
        "num_samples": int(totals["num_samples"]),
    }


def run_epoch(
    model: SST5ContentAE,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    split_name: str = "train",
) -> Dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = empty_totals()

    for step, batch in enumerate(loader, start=1):
        input_logitc_K = batch["input_logitc_K"].to(device, non_blocking=True).float()
        target_k_indices = batch["target_k_indices"].to(device, non_blocking=True).long()
        loss_mask = batch["loss_mask"].to(device, non_blocking=True).bool()
        target_onehot_K = batch.get("target_onehot_K")
        if target_onehot_K is not None:
            target_onehot_K = target_onehot_K.to(device, non_blocking=True)
        batch_size = int(target_k_indices.shape[0])

        with torch.set_grad_enabled(is_train):
            outputs = model(input_logitc_K)
            losses = compute_k_space_loss(
                latent_logits_K=outputs["latent_logits"],
                recon_logitc_K=outputs["recon_logits"],
                input_logitc_K=input_logitc_K,
                target_k_indices=target_k_indices,
                target_onehot_K=target_onehot_K,
                loss_mask=loss_mask,
                ce_weight=args.ce_weight,
                kl_weight=args.kl_weight,
                temperature=args.temperature,
                verify_target_onehot=args.verify_target_onehot,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()

        token_total = float(losses["token_total"].item())
        totals["loss"] += float(losses["loss"].detach().item()) * token_total
        totals["loss_ce"] += float(losses["loss_ce"].detach().item()) * token_total
        totals["loss_kl"] += float(losses["loss_kl"].detach().item()) * token_total
        totals["token_correct"] += float(losses["token_correct"].item())
        totals["token_total"] += token_total
        totals["num_samples"] += batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            partial = finalize_totals(totals)
            LOGGER.info(
                "%s step=%d samples=%d tokens=%d %s/loss=%.6f %s/loss_ce=%.6f "
                "%s/loss_kl=%.6f %s/token_acc_K=%.4f",
                split_name,
                step,
                partial["num_samples"],
                partial["token_total"],
                split_name,
                partial["loss"],
                split_name,
                partial["loss_ce"],
                split_name,
                partial["loss_kl"],
                split_name,
                partial["token_acc_K"],
            )

    return finalize_totals(totals)


def write_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    model: SST5ContentAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    train_metrics: Dict,
    val_metrics: Dict,
    cache_metadata: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "config": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "cache_metadata": cache_metadata,
        },
        path,
    )


def build_loader(dataset: SST5TeacherLogitsDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator if shuffle else None,
        collate_fn=sst5_k_space_collate_fn,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    logits_dtype = dtype_from_name(args.logits_dtype)

    train_dataset = SST5TeacherLogitsDataset(
        args.train_cache,
        expected_num_labels=args.expected_num_labels,
        logits_dtype=logits_dtype,
    )
    val_dataset = SST5TeacherLogitsDataset(
        args.val_cache,
        expected_num_labels=args.expected_num_labels,
        logits_dtype=logits_dtype,
    )
    if train_dataset.k != val_dataset.k:
        raise ValueError(f"Train K={train_dataset.k} does not match val K={val_dataset.k}")
    if train_dataset.candidate_token_ids != val_dataset.candidate_token_ids:
        raise ValueError("Train and val frozen K token ids differ")

    model = SST5ContentAE(
        SST5ContentAEConfig(
            k=train_dataset.k,
            fusion_hidden_dim=args.fusion_hidden_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            dropout=args.dropout,
            residual_fusion=not args.no_residual_fusion,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader = build_loader(val_dataset, args, shuffle=False)
    config_out = {
        "args": vars(args),
        "model_config": model.config.to_dict(),
        "cache_format": "token_sequence_k_space_v1",
        "k": train_dataset.k,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_tokens": train_dataset.total_tokens,
        "val_tokens": val_dataset.total_tokens,
        "candidate_token_ids": train_dataset.candidate_token_ids,
        "train_cache_metadata": train_dataset.metadata,
        "val_cache_metadata": val_dataset.metadata,
        "loss_formula": (
            "loss = 0.5 * CE(latent_logits_K.reshape(-1,K), target_k_indices.reshape(-1)) "
            "+ 0.5 * KL(recon_logitc_K, input_logitc_K), with loss_mask over [B,T]."
        ),
    }
    write_json(output_dir / "train_config.json", config_out)

    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    LOGGER.info(
        "Training SST-5 frozen K-space AE | K=%d train=%d/%d tokens val=%d/%d tokens device=%s",
        train_dataset.k,
        len(train_dataset),
        train_dataset.total_tokens,
        len(val_dataset),
        val_dataset.total_tokens,
        device,
    )
    for epoch in range(1, args.epochs + 1):
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
                split_name="valid",
            )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": val_metrics,
        }
        write_jsonl(progress_path, row)
        LOGGER.info(
            "epoch=%d train/loss=%.6f train/loss_ce=%.6f train/loss_kl=%.6f "
            "train/token_acc_K=%.4f valid/loss=%.6f valid/loss_ce=%.6f "
            "valid/loss_kl=%.6f valid/token_acc_K=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["loss_ce"],
            train_metrics["loss_kl"],
            train_metrics["token_acc_K"],
            val_metrics["loss"],
            val_metrics["loss_ce"],
            val_metrics["loss_kl"],
            val_metrics["token_acc_K"],
        )

        improved = (
            val_metrics["token_acc_K"] > best_val_acc
            or (
                val_metrics["token_acc_K"] == best_val_acc
                and val_metrics["loss"] < best_val_loss
            )
        )
        if improved:
            best_val_acc = float(val_metrics["token_acc_K"])
            best_val_loss = float(val_metrics["loss"])
            save_checkpoint(
                output_dir / "best_sst5_content_ae.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                cache_metadata=train_dataset.metadata,
            )

    save_checkpoint(
        output_dir / "final_sst5_content_ae.pt",
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        args=args,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        cache_metadata=train_dataset.metadata,
    )
    write_json(
        output_dir / "final_metrics.json",
        {
            "best_valid_token_acc_K": best_val_acc,
            "best_valid_loss": best_val_loss,
            "last_train": train_metrics,
            "last_valid": val_metrics,
        },
    )


if __name__ == "__main__":
    main()
