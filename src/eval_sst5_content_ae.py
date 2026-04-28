from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sst5_common import load_pt, project_path, write_json
from sst5_dataset import SST5TeacherLogitsDataset, sst5_k_space_collate_fn
from sst5_content_ae_model import SST5ContentAE, SST5ContentAEConfig
from train_sst5_content_ae import (
    compute_k_space_loss,
    dtype_from_name,
    empty_totals,
    finalize_totals,
    resolve_device,
)


LOGGER = logging.getLogger("eval_sst5_content_ae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SST-5 frozen K-space AE.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--expected-num-labels", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--logits-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[SST5ContentAE, Dict]:
    checkpoint = load_pt(checkpoint_path)
    cfg = checkpoint["model_config"]
    model = SST5ContentAE(SST5ContentAEConfig(**cfg))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def per_token_kl(
    input_logitc_K: torch.Tensor,
    recon_logitc_K: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    teacher_probs_K = F.softmax(input_logitc_K.float() / temperature, dim=-1)
    student_log_probs_K = F.log_softmax(recon_logitc_K.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs_K, teacher_probs_K, reduction="none").sum(dim=-1) * (
        temperature**2
    )


def top_token_entries(
    logits: torch.Tensor,
    candidate_texts: List[str],
    top_k: int = 8,
) -> List[Dict]:
    search_logits = logits.float()
    k_eff = min(int(top_k), int(search_logits.numel()))
    values, indices = search_logits.topk(k_eff)
    entries: List[Dict] = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        idx = int(index)
        text = candidate_texts[idx] if idx < len(candidate_texts) else ""
        entries.append({"k_index": idx, "text": text, "logit": float(value)})
    return entries


def write_sample(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = SST5TeacherLogitsDataset(
        args.cache,
        expected_num_labels=args.expected_num_labels,
        logits_dtype=dtype_from_name(args.logits_dtype),
    )
    candidate_texts = [str(v) for v in dataset.metadata.get("candidate_texts", [])]
    model, checkpoint = load_model(args.checkpoint, device=device)
    if model.k != dataset.k:
        raise ValueError(f"Checkpoint K={model.k} does not match cache K={dataset.k}")
    train_config = checkpoint.get("config", {})
    temperature = float(args.temperature if args.temperature is not None else train_config.get("temperature", 1.0))
    ce_weight = float(train_config.get("ce_weight", train_config.get("lambda_ce", 0.5)))
    kl_weight = float(train_config.get("kl_weight", train_config.get("lambda_kl", 0.5)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=sst5_k_space_collate_fn,
    )
    totals = empty_totals()
    sample_path = output_dir / "eval_predictions_sample.jsonl"
    if sample_path.exists():
        sample_path.unlink()
    samples_written = 0

    with torch.no_grad():
        for batch in loader:
            input_logitc_K = batch["input_logitc_K"].to(device, non_blocking=True).float()
            target_k_indices = batch["target_k_indices"].to(device, non_blocking=True).long()
            loss_mask = batch["loss_mask"].to(device, non_blocking=True).bool()
            outputs = model(input_logitc_K)
            losses = compute_k_space_loss(
                latent_logits_K=outputs["latent_logits"],
                recon_logitc_K=outputs["recon_logits"],
                input_logitc_K=input_logitc_K,
                target_k_indices=target_k_indices,
                loss_mask=loss_mask,
                ce_weight=ce_weight,
                kl_weight=kl_weight,
                temperature=temperature,
                verify_target_onehot=True,
            )

            token_total = float(losses["token_total"].item())
            totals["loss"] += float(losses["loss"].item()) * token_total
            totals["loss_ce"] += float(losses["loss_ce"].item()) * token_total
            totals["loss_kl"] += float(losses["loss_kl"].item()) * token_total
            totals["token_correct"] += float(losses["token_correct"].item())
            totals["token_total"] += token_total
            totals["num_samples"] += int(target_k_indices.shape[0])

            if samples_written < args.sample_size:
                token_kl = per_token_kl(input_logitc_K, outputs["recon_logits"], temperature)
                pred_k = outputs["latent_logits"].argmax(dim=-1)
                rows_to_write = min(args.sample_size - samples_written, int(target_k_indices.shape[0]))
                for i in range(rows_to_write):
                    valid_positions = torch.nonzero(loss_mask[i], as_tuple=False).view(-1)
                    token_rows = []
                    for pos in valid_positions[:8].detach().cpu().tolist():
                        target_k = int(target_k_indices[i, pos].item())
                        pred = int(pred_k[i, pos].item())
                        token_rows.append(
                            {
                                "position": int(pos),
                                "teacher_next_token_id": int(batch["teacher_next_token_ids"][i, pos].item()),
                                "target_k_index": target_k,
                                "target_text": candidate_texts[target_k] if target_k < len(candidate_texts) else "",
                                "pred_k_index": pred,
                                "pred_text": candidate_texts[pred] if pred < len(candidate_texts) else "",
                                "token_kl": float(token_kl[i, pos].item()),
                                "input_top_tokens": top_token_entries(
                                    input_logitc_K[i, pos].detach().cpu(),
                                    candidate_texts,
                                ),
                                "latent_top_tokens": top_token_entries(
                                    outputs["latent_logits"][i, pos].detach().cpu(),
                                    candidate_texts,
                                ),
                                "recon_top_tokens": top_token_entries(
                                    outputs["recon_logits"][i, pos].detach().cpu(),
                                    candidate_texts,
                                ),
                            }
                        )
                    write_sample(
                        sample_path,
                        {
                            "index": int(batch["index"][i].item()),
                            "text": batch["text"][i],
                            "label": int(batch["label"][i].item()),
                            "label_text": batch["label_text"][i],
                            "teacher_cot_output": batch["teacher_cot_output"][i],
                            "tokens": token_rows,
                        },
                    )
                samples_written += rows_to_write

    metrics = finalize_totals(totals)
    metrics.update(
        {
            "split": args.split_name,
            "cache": str(project_path(args.cache)),
            "checkpoint": str(project_path(args.checkpoint)),
            "temperature": temperature,
            "ce_weight": ce_weight,
            "kl_weight": kl_weight,
            "k": dataset.k,
            "num_samples": len(dataset),
            "sample_path": str(sample_path),
            "cache_format": dataset.metadata.get("cache_format"),
        }
    )
    write_json(output_dir / "eval_metrics.json", metrics)
    LOGGER.info(
        "split=%s samples=%d tokens=%d loss=%.6f loss_ce=%.6f loss_kl=%.6f token_acc_K=%.4f",
        args.split_name,
        metrics["num_samples"],
        metrics["token_total"],
        metrics["loss"],
        metrics["loss_ce"],
        metrics["loss_kl"],
        metrics["token_acc_K"],
    )


if __name__ == "__main__":
    main()
