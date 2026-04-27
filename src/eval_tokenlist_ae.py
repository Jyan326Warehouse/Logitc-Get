import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gsm_tokenlist_dataset import GSMTokenListDataset, load_token_list
from tokenlist_ae_model import GSMTokenListAE, GSMTokenListAEConfig


LOGGER = logging.getLogger("eval_tokenlist_ae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM token-list aligned AE.")
    parser.add_argument("--logits-dir", required=True)
    parser.add_argument("--token-list-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def topk_correct(logits: torch.Tensor, target_idx: torch.Tensor, k: int) -> int:
    k_eff = min(int(k), logits.shape[-1])
    pred = logits.topk(k_eff, dim=-1).indices
    return int(pred.eq(target_idx.view(-1, 1)).any(dim=-1).sum().item())


def rank_sum(logits: torch.Tensor, target_idx: torch.Tensor) -> float:
    true_scores = logits.gather(dim=-1, index=target_idx.view(-1, 1))
    ranks = (logits > true_scores).sum(dim=-1).to(torch.float32) + 1.0
    return float(ranks.sum().item())


def per_sample_losses(
    input_logits_k: torch.Tensor,
    target_idx: torch.Tensor,
    latent_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    p = F.softmax(input_logits_k / temperature, dim=-1)
    log_q = F.log_softmax(recon_logits / temperature, dim=-1)
    kl = F.kl_div(log_q, p, reduction="none").sum(dim=-1) * (temperature * temperature)
    latent_ce = F.cross_entropy(latent_logits, target_idx, reduction="none")
    recon_ce = F.cross_entropy(recon_logits, target_idx, reduction="none")
    return {
        "kl": kl,
        "latent_ce": latent_ce,
        "recon_ce": recon_ce,
    }


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_token_text_lookup(token_list: Dict) -> Dict[int, str]:
    token_ids = [int(t) for t in token_list["token_ids"]]
    token_texts = token_list.get("token_texts", [""] * len(token_ids))
    if len(token_texts) < len(token_ids):
        token_texts = token_texts + [""] * (len(token_ids) - len(token_texts))
    return {token_id: str(token_texts[idx]) for idx, token_id in enumerate(token_ids)}


def top_indices_to_token_ids(top_indices: List[int], token_ids: List[int]) -> List[int]:
    return [int(token_ids[int(idx)]) for idx in top_indices]


def token_ids_to_texts(ids: List[int], token_text_by_id: Dict[int, str]) -> List[str]:
    return [token_text_by_id.get(int(token_id), "") for token_id in ids]


def load_model_from_checkpoint(
    checkpoint_path: str,
    k: int,
    device: torch.device,
) -> tuple[GSMTokenListAE, Dict]:
    checkpoint = load_pt(Path(checkpoint_path))
    model_config_dict = checkpoint.get("model_config", {})
    train_config = checkpoint.get("config", {})
    hidden_dim = int(model_config_dict.get("hidden_dim", train_config.get("hidden_dim", 1024)))
    dropout = float(model_config_dict.get("dropout", train_config.get("dropout", 0.1)))

    model = GSMTokenListAE(
        GSMTokenListAEConfig(
            k=int(model_config_dict.get("k", train_config.get("k", k))),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    )
    if model.k != k:
        raise ValueError(f"Checkpoint K={model.k} does not match token list K={k}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_list = load_token_list(args.token_list_json)
    token_ids = [int(t) for t in token_list["token_ids"]]
    token_text_by_id = build_token_text_lookup(token_list)
    k = len(token_ids)
    device = resolve_device(args.device)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, k=k, device=device)
    train_config = checkpoint.get("config", {})
    temperature = float(
        args.temperature
        if args.temperature is not None
        else train_config.get("temperature", 1.0)
    )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    dataset = GSMTokenListDataset(
        logits_dir=args.logits_dir,
        token_list_json=args.token_list_json,
        skip_oov=True,
        preload=args.preload,
    )
    if len(dataset) == 0:
        raise ValueError("Eval dataset has zero evaluated positions.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    LOGGER.info(
        "Evaluating split=%s | samples=%d total_positions=%d skipped_oov=%d oov_rate=%.6f",
        args.split_name,
        len(dataset),
        dataset.total_positions,
        dataset.skipped_oov_count,
        dataset.oov_rate,
    )

    totals = {
        "kl": 0.0,
        "latent_ce": 0.0,
        "recon_ce": 0.0,
        "latent_top1_acc": 0.0,
        "latent_top5_acc": 0.0,
        "latent_top10_acc": 0.0,
        "recon_top1_acc": 0.0,
        "recon_top5_acc": 0.0,
        "recon_top10_acc": 0.0,
        "latent_rank_sum": 0.0,
        "recon_rank_sum": 0.0,
        "num_samples": 0.0,
    }

    sample_path = output_dir / "eval_predictions_sample.jsonl"
    if sample_path.exists():
        sample_path.unlink()
    samples_written = 0

    with torch.no_grad():
        for batch in loader:
            input_logits_k = batch["input_logits_k"].to(device, non_blocking=True).float()
            target_idx = batch["target_idx"].to(device, non_blocking=True).long()
            outputs = model(input_logits_k)
            latent_logits = outputs["latent_logits"]
            recon_logits = outputs["recon_logits"]
            losses = per_sample_losses(
                input_logits_k=input_logits_k,
                target_idx=target_idx,
                latent_logits=latent_logits,
                recon_logits=recon_logits,
                temperature=temperature,
            )
            batch_size = int(target_idx.numel())

            totals["kl"] += float(losses["kl"].sum().item())
            totals["latent_ce"] += float(losses["latent_ce"].sum().item())
            totals["recon_ce"] += float(losses["recon_ce"].sum().item())
            totals["latent_top1_acc"] += topk_correct(latent_logits, target_idx, 1)
            totals["latent_top5_acc"] += topk_correct(latent_logits, target_idx, 5)
            totals["latent_top10_acc"] += topk_correct(latent_logits, target_idx, 10)
            totals["recon_top1_acc"] += topk_correct(recon_logits, target_idx, 1)
            totals["recon_top5_acc"] += topk_correct(recon_logits, target_idx, 5)
            totals["recon_top10_acc"] += topk_correct(recon_logits, target_idx, 10)
            totals["latent_rank_sum"] += rank_sum(latent_logits, target_idx)
            totals["recon_rank_sum"] += rank_sum(recon_logits, target_idx)
            totals["num_samples"] += batch_size

            if samples_written < args.sample_size:
                sample_topk = min(10, k)
                latent_top = latent_logits.topk(sample_topk, dim=-1).indices.cpu().tolist()
                recon_top = recon_logits.topk(sample_topk, dim=-1).indices.cpu().tolist()
                target_idx_cpu = target_idx.cpu().tolist()
                token_id_cpu = batch["token_id"].cpu().tolist()
                position_cpu = batch["position"].cpu().tolist()
                source_files = batch["source_file"]
                kl_cpu = losses["kl"].cpu().tolist()
                latent_ce_cpu = losses["latent_ce"].cpu().tolist()
                recon_ce_cpu = losses["recon_ce"].cpu().tolist()

                for row_idx in range(batch_size):
                    if samples_written >= args.sample_size:
                        break
                    true_token_id = int(token_id_cpu[row_idx])
                    latent_ids = top_indices_to_token_ids(latent_top[row_idx], token_ids)
                    recon_ids = top_indices_to_token_ids(recon_top[row_idx], token_ids)
                    sample_record = {
                        "source_file": source_files[row_idx],
                        "position": int(position_cpu[row_idx]),
                        "true_token_id": true_token_id,
                        "true_token_text": token_text_by_id.get(true_token_id, ""),
                        "target_idx": int(target_idx_cpu[row_idx]),
                        "latent_top10_token_ids": latent_ids,
                        "latent_top10_token_texts": token_ids_to_texts(latent_ids, token_text_by_id),
                        "recon_top10_token_ids": recon_ids,
                        "recon_top10_token_texts": token_ids_to_texts(recon_ids, token_text_by_id),
                        "kl": float(kl_cpu[row_idx]),
                        "latent_ce": float(latent_ce_cpu[row_idx]),
                        "recon_ce": float(recon_ce_cpu[row_idx]),
                    }
                    append_jsonl(sample_path, sample_record)
                    samples_written += 1

    n = max(int(totals["num_samples"]), 1)
    metrics = {
        "split_name": args.split_name,
        "checkpoint": args.checkpoint,
        "token_list_json": args.token_list_json,
        "k": k,
        "temperature": temperature,
        "reconstruction_kl": totals["kl"] / n,
        "latent_ce": totals["latent_ce"] / n,
        "recon_ce": totals["recon_ce"] / n,
        "latent_top1_acc": totals["latent_top1_acc"] / n,
        "latent_top5_acc": totals["latent_top5_acc"] / n,
        "latent_top10_acc": totals["latent_top10_acc"] / n,
        "recon_top1_acc": totals["recon_top1_acc"] / n,
        "recon_top5_acc": totals["recon_top5_acc"] / n,
        "recon_top10_acc": totals["recon_top10_acc"] / n,
        "latent_true_token_mean_rank": totals["latent_rank_sum"] / n,
        "recon_true_token_mean_rank": totals["recon_rank_sum"] / n,
        "total_positions": int(dataset.total_positions),
        "evaluated_positions": int(len(dataset)),
        "skipped_oov_positions": int(dataset.skipped_oov_count),
        "oov_rate": float(dataset.oov_rate),
        "samples_written": int(samples_written),
    }
    metrics_path = output_dir / "eval_metrics.json"
    write_json(metrics_path, metrics)
    LOGGER.info("Wrote %s", metrics_path)
    LOGGER.info("Wrote %s", sample_path)


if __name__ == "__main__":
    main()
