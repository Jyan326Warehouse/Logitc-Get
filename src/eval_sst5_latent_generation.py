from __future__ import annotations

"""
Example:

python src/eval_sst5_latent_generation.py `
  --model-path models/Qwen3-8B `
  --checkpoint outputs/sst5_kspace_seq_qwen3_lr5e-5/best_sst5_content_ae.pt `
  --test-cache data/sst5/teacher_logits/Qwen3-8B_kspace_seq/test.pt `
  --output-dir outputs/sst5_kspace_seq_qwen3_lr5e-5/latent_generation_eval `
  --max-new-tokens 64 `
  --temperature 1.0 `
  --trust-remote-code
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sst5_common import (
    SST5LabelSpace,
    build_sst5_cot_prompt,
    extract_final_label_from_text,
    load_pt,
    project_path,
    resolve_dtype,
    write_json,
)
from sst5_content_ae_model import SST5ContentAE, SST5ContentAEConfig


LOGGER = logging.getLogger("eval_sst5_latent_generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SST-5 generation with AE latent_logits replacing teacher K-space "
            "next-token logits at test time."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-generations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-teacher-k-baseline", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def model_load_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return resolve_dtype(dtype_name)


def normalize_label_text_map(raw: Dict) -> Dict[int, str]:
    return {int(k): str(v) for k, v in raw.items()}


def load_generation_cache_metadata(test_cache: str | Path, max_samples: Optional[int]) -> Dict:
    obj = load_pt(test_cache)
    required = (
        "texts",
        "labels",
        "label_texts",
        "candidate_token_ids",
        "candidate_texts",
        "label_values",
        "label_text_by_value",
        "k",
    )
    for key in required:
        if key not in obj:
            raise KeyError(f"Test cache {test_cache} missing key {key!r}")

    texts = [str(v) for v in obj["texts"]]
    labels_tensor = obj["labels"]
    labels = (
        labels_tensor.detach().cpu().to(dtype=torch.long).tolist()
        if torch.is_tensor(labels_tensor)
        else [int(v) for v in labels_tensor]
    )
    label_texts = [str(v) for v in obj["label_texts"]]
    if not (len(texts) == len(labels) == len(label_texts)):
        raise ValueError(
            "Test cache metadata lengths differ: "
            f"texts={len(texts)} labels={len(labels)} label_texts={len(label_texts)}"
        )
    if max_samples is not None:
        texts = texts[: int(max_samples)]
        labels = labels[: int(max_samples)]
        label_texts = label_texts[: int(max_samples)]

    candidate_token_ids = [int(v) for v in obj["candidate_token_ids"]]
    candidate_texts = [str(v) for v in obj["candidate_texts"]]
    k = int(obj["k"])
    if len(candidate_token_ids) != k:
        raise ValueError(f"candidate_token_ids length {len(candidate_token_ids)} does not match K={k}")
    if len(candidate_texts) != k:
        raise ValueError(f"candidate_texts length {len(candidate_texts)} does not match K={k}")

    metadata = {
        "texts": texts,
        "labels": [int(v) for v in labels],
        "label_texts": label_texts,
        "candidate_token_ids": candidate_token_ids,
        "candidate_texts": candidate_texts,
        "label_values": [int(v) for v in obj["label_values"]],
        "label_text_by_value": normalize_label_text_map(obj["label_text_by_value"]),
        "k": k,
        "source_cache": str(project_path(test_cache)),
        "cache_format": obj.get("cache_format"),
    }
    del obj
    gc.collect()
    return metadata


def load_teacher(model_path: str, args: argparse.Namespace, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad_token or eos_token; set one before generation.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_load_dtype(args.load_dtype, device),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def load_ae_model(checkpoint_path: str | Path, device: torch.device, expected_k: int) -> SST5ContentAE:
    checkpoint = load_pt(checkpoint_path)
    cfg = checkpoint["model_config"]
    model = SST5ContentAE(SST5ContentAEConfig(**cfg))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    if model.k != int(expected_k):
        raise ValueError(f"AE checkpoint K={model.k} does not match test cache K={expected_k}")
    return model


def apply_top_k_top_p(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None:
        k = max(1, min(int(top_k), int(filtered.numel())))
        cutoff = torch.topk(filtered, k).values[..., -1]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
    if top_p is not None:
        p = float(top_p)
        if not 0.0 < p <= 1.0:
            raise ValueError(f"top_p must be in (0,1], got {top_p}")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered[sorted_indices[remove]] = float("-inf")
    return filtered


def choose_next_k(
    logits_K: torch.Tensor,
    temperature: float,
    do_sample: bool,
    top_k: Optional[int],
    top_p: Optional[float],
) -> int:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = logits_K.float() / float(temperature)
    scaled = apply_top_k_top_p(scaled, top_k=top_k, top_p=top_p)
    if do_sample:
        probs = F.softmax(scaled, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
            raise ValueError("Sampling distribution became invalid after top-k/top-p filtering")
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.argmax(scaled).item())


def next_full_logits(model, current_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=current_ids)
    return outputs.logits[:, -1, :]


def generate_teacher_baseline(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_k: Optional[int],
    top_p: Optional[float],
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        if top_k is not None:
            generate_kwargs["top_k"] = int(top_k)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)
    with torch.no_grad():
        generated = model.generate(**encoded, **generate_kwargs)
    new_ids = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_k_space(
    model,
    tokenizer,
    ae_model: Optional[SST5ContentAE],
    prompt: str,
    candidate_token_ids: List[int],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_k: Optional[int],
    top_p: Optional[float],
    use_latent: bool,
) -> Dict:
    encoded = tokenizer(prompt, return_tensors="pt")
    current_ids = encoded["input_ids"].to(device)
    candidate_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device)
    generated_token_ids: List[int] = []
    generated_k_indices: List[int] = []

    for _ in range(int(max_new_tokens)):
        full_logits = next_full_logits(model, current_ids)
        input_logitc_K = full_logits.index_select(dim=-1, index=candidate_ids)
        if use_latent:
            if ae_model is None:
                raise ValueError("ae_model is required for latent generation")
            ae_dtype = next(ae_model.parameters()).dtype
            ae_input = input_logitc_K.to(dtype=ae_dtype).unsqueeze(1)
            with torch.no_grad():
                outputs = ae_model(ae_input)
            next_logits_K = outputs["latent_logits"][:, -1, :]
            # Intentionally never use outputs["recon_logits"] for generation.
        else:
            next_logits_K = input_logitc_K

        next_k = choose_next_k(
            logits_K=next_logits_K[0],
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
        )
        next_token_id = int(candidate_token_ids[next_k])
        generated_k_indices.append(next_k)
        generated_token_ids.append(next_token_id)
        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        current_ids = torch.cat([current_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token_id == int(tokenizer.eos_token_id):
            break

    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_token_ids": generated_token_ids,
        "generated_k_indices": generated_k_indices,
    }


def accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total > 0 else 0.0


def write_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "generation_predictions.jsonl"
    if prediction_path.exists() and args.save_generations:
        prediction_path.unlink()

    device = resolve_device(args.device)
    cache = load_generation_cache_metadata(args.test_cache, max_samples=args.max_samples)
    label_space = SST5LabelSpace(
        label_values=cache["label_values"],
        label_text_by_value=cache["label_text_by_value"],
        expected_num_labels=len(cache["label_values"]),
        num_train_records=0,
    )
    tokenizer, teacher_model = load_teacher(args.model_path, args, device=device)
    ae_model = load_ae_model(args.checkpoint, device=device, expected_k=cache["k"])

    teacher_correct = 0
    latent_correct = 0
    teacher_k_correct = 0
    total = len(cache["texts"])
    LOGGER.info(
        "Starting latent generation eval | samples=%d K=%d device=%s max_new_tokens=%d",
        total,
        cache["k"],
        device,
        args.max_new_tokens,
    )

    for idx, (text, label, label_text) in enumerate(
        tqdm(
            zip(cache["texts"], cache["labels"], cache["label_texts"]),
            total=total,
            desc="latent generation",
        )
    ):
        prompt = build_sst5_cot_prompt(text, label_space=label_space)

        teacher_output = generate_teacher_baseline(
            model=teacher_model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        teacher_pred = extract_final_label_from_text(teacher_output, cache["label_values"])
        if teacher_pred == int(label):
            teacher_correct += 1

        latent = generate_k_space(
            model=teacher_model,
            tokenizer=tokenizer,
            ae_model=ae_model,
            prompt=prompt,
            candidate_token_ids=cache["candidate_token_ids"],
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            use_latent=True,
        )
        latent_pred = extract_final_label_from_text(latent["text"], cache["label_values"])
        if latent_pred == int(label):
            latent_correct += 1

        teacher_k = None
        teacher_k_pred = None
        if args.run_teacher_k_baseline:
            teacher_k = generate_k_space(
                model=teacher_model,
                tokenizer=tokenizer,
                ae_model=None,
                prompt=prompt,
                candidate_token_ids=cache["candidate_token_ids"],
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                use_latent=False,
            )
            teacher_k_pred = extract_final_label_from_text(teacher_k["text"], cache["label_values"])
            if teacher_k_pred == int(label):
                teacher_k_correct += 1

        if args.save_generations:
            row = {
                "index": idx,
                "text": text,
                "label": int(label),
                "label_text": label_text,
                "teacher_output": teacher_output,
                "teacher_pred": teacher_pred,
                "teacher_correct": teacher_pred == int(label),
                "latent_output": latent["text"],
                "latent_pred": latent_pred,
                "latent_correct": latent_pred == int(label),
                "latent_generated_token_ids": latent["generated_token_ids"],
                "latent_generated_k_indices": latent["generated_k_indices"],
            }
            if args.run_teacher_k_baseline and teacher_k is not None:
                row.update(
                    {
                        "teacher_k_output": teacher_k["text"],
                        "teacher_k_pred": teacher_k_pred,
                        "teacher_k_correct": teacher_k_pred == int(label),
                        "teacher_k_generated_token_ids": teacher_k["generated_token_ids"],
                        "teacher_k_generated_k_indices": teacher_k["generated_k_indices"],
                    }
                )
            write_jsonl(prediction_path, row)

    metrics = {
        "model_path": args.model_path,
        "checkpoint": str(project_path(args.checkpoint)),
        "test_cache": cache["source_cache"],
        "output_dir": str(output_dir),
        "num_samples": total,
        "k": cache["k"],
        "label_values": cache["label_values"],
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "do_sample": bool(args.do_sample),
        "top_k": None if args.top_k is None else int(args.top_k),
        "top_p": None if args.top_p is None else float(args.top_p),
        "teacher_baseline_correct": int(teacher_correct),
        "teacher_baseline_accuracy": accuracy(teacher_correct, total),
        "latent_correct": int(latent_correct),
        "latent_accuracy": accuracy(latent_correct, total),
        "save_generations": bool(args.save_generations),
        "generation_predictions_path": str(prediction_path) if args.save_generations else None,
        "cache_format": cache["cache_format"],
        "uses_recon_logits_for_generation": False,
    }
    if args.run_teacher_k_baseline:
        metrics.update(
            {
                "teacher_k_baseline_correct": int(teacher_k_correct),
                "teacher_k_baseline_accuracy": accuracy(teacher_k_correct, total),
            }
        )
    write_json(output_dir / "generation_metrics.json", metrics)
    LOGGER.info(
        "generation eval complete | samples=%d teacher_acc=%.4f latent_acc=%.4f",
        total,
        metrics["teacher_baseline_accuracy"],
        metrics["latent_accuracy"],
    )


if __name__ == "__main__":
    main()
