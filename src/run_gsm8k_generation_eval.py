import argparse
import json
import logging
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm_tokenlist_dataset import load_token_list
from tokenlist_ae_model import GSMTokenListAE, GSMTokenListAEConfig


LOGGER = logging.getLogger("run_gsm8k_generation_eval")

PROMPT_TEMPLATE = """Solve the following math problem step by step.

Question:
{question}

Answer:
"""

FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GSM8K generation baselines with optional token-list AE latent "
            "logit intervention. This script does not train or fine-tune the LLM."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-meta", default="data/meta/main_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/gsm8k_generation_eval")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "ae_latent"],
        choices=[
            "baseline",
            "ae_latent",
            "ae_recon",
            "ae_latent_patch",
            "ae_recon_patch",
        ],
        help=(
            "baseline uses original full-vocab logits. ae_latent/ae_recon decode "
            "only inside T_GSM. *_patch keeps full vocab but replaces T_GSM logits "
            "with AE latent/recon logits before softmax."
        ),
    )
    parser.add_argument("--token-list-json", default="outputs/gsm_token_list_all_text_rebuilt_4000/gsm_token_list.json")
    parser.add_argument("--checkpoint", default="outputs/tokenlist_ae_all_text_rebuilt_4000/best_tokenlist_ae.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--decode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--ae-device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--ae-logit-scale",
        type=float,
        default=1.0,
        help="Scale applied to latent/recon logits before decoding or patching.",
    )
    parser.add_argument(
        "--ae-logit-bias",
        type=float,
        default=0.0,
        help="Bias applied to latent/recon logits before decoding or patching.",
    )
    parser.add_argument(
        "--outside-token-bias",
        type=float,
        default=0.0,
        help=(
            "Only used by *_patch modes. Adds this bias to tokens outside T_GSM "
            "before replacing T_GSM logits with AE logits. Negative values make "
            "outside-vocab tokens less likely."
        ),
    )
    parser.add_argument(
        "--stop-after-final-answer",
        action="store_true",
        help="Stop early once generated text contains a GSM8K-style '#### number'.",
    )
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


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_arg]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, max_samples: Optional[int]) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx}: {path}") from exc
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def build_prompt(record: Dict) -> str:
    prompt_text = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    if prompt_text:
        return prompt_text

    question = str(record.get("question") or record.get("input") or "").strip()
    if question:
        return PROMPT_TEMPLATE.format(question=question)

    text = str(record.get("text") or "").strip()
    if text:
        return text

    raise KeyError("Record is missing prompt_text/prompt/question/input/text")


def record_gold_text(record: Dict) -> str:
    for key in ("final_answer", "answer_text", "answer", "output"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        return final_matches[-1]

    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return number_matches[-1]

    return None


def decimal_or_none(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "")
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def normalize_answer(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    dec = decimal_or_none(value)
    if dec is not None:
        normalized = format(dec.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized
    return str(value).strip().replace(",", "").rstrip(".")


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_dec = decimal_or_none(pred)
    gold_dec = decimal_or_none(gold)
    if pred_dec is not None and gold_dec is not None:
        return pred_dec == gold_dec
    return normalize_answer(pred) == normalize_answer(gold)


def get_sample_id(record: Dict, idx: int) -> str:
    return str(record.get("sample_id") or record.get("id") or f"sample_{idx:06d}")


def model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eos_token_ids(tokenizer, model) -> set[int]:
    ids: set[int] = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(int(value))
        elif isinstance(value, Iterable):
            ids.update(int(item) for item in value if item is not None)
    return ids


def load_ae_model(
    checkpoint_path: str,
    k: int,
    device: torch.device,
) -> Tuple[GSMTokenListAE, Dict]:
    checkpoint = load_pt(Path(checkpoint_path))
    model_config_dict = checkpoint.get("model_config", {})
    train_config = checkpoint.get("config", {})

    config = GSMTokenListAEConfig(
        k=int(model_config_dict.get("k", train_config.get("k", k))),
        hidden_dim=int(model_config_dict.get("hidden_dim", train_config.get("hidden_dim", 1024))),
        dropout=float(model_config_dict.get("dropout", train_config.get("dropout", 0.1))),
    )
    if config.k != k:
        raise ValueError(f"Checkpoint K={config.k} does not match token list K={k}")

    model = GSMTokenListAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def mode_needs_ae(mode: str) -> bool:
    return mode.startswith("ae_")


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.float().clone()

    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = -float("inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        if remove.numel() > 1:
            remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = -float("inf")

    return filtered


def choose_index_from_logits(
    logits: torch.Tensor,
    decode: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    logits = logits.float().view(-1)
    if decode == "greedy":
        return int(torch.argmax(logits).item())

    if temperature <= 0:
        raise ValueError(f"temperature must be positive for sampling, got {temperature}")

    filtered = top_k_top_p_filter(logits / temperature, top_k=top_k, top_p=top_p)
    if torch.isneginf(filtered).all():
        return int(torch.argmax(logits).item())
    probs = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def ae_logits_for_step(
    base_logits: torch.Tensor,
    ae_model: GSMTokenListAE,
    token_ids_model_device: torch.Tensor,
    ae_device: torch.device,
    mode: str,
    ae_logit_scale: float,
    ae_logit_bias: float,
) -> torch.Tensor:
    input_logits_k = base_logits.index_select(0, token_ids_model_device).to(ae_device).float().unsqueeze(0)
    outputs = ae_model(input_logits_k)
    if "latent" in mode:
        logits_k = outputs["latent_logits"][0]
    else:
        logits_k = outputs["recon_logits"][0]
    return logits_k.float() * float(ae_logit_scale) + float(ae_logit_bias)


def select_next_token_id(
    base_logits: torch.Tensor,
    mode: str,
    args: argparse.Namespace,
    token_ids: Sequence[int],
    token_ids_model_device: Optional[torch.Tensor],
    ae_model: Optional[GSMTokenListAE],
    ae_device: Optional[torch.device],
) -> Tuple[int, Dict]:
    base_logits = base_logits.float().view(-1)

    if mode == "baseline":
        next_id = choose_index_from_logits(
            base_logits,
            decode=args.decode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return next_id, {
            "selection_space": "full_vocab",
            "selected_tokenlist_index": None,
        }

    if token_ids_model_device is None or ae_model is None or ae_device is None:
        raise ValueError(f"Mode {mode} requires token list and AE checkpoint.")
    token_ids_for_logits = token_ids_model_device
    if token_ids_for_logits.device != base_logits.device:
        token_ids_for_logits = token_ids_for_logits.to(base_logits.device)

    logits_k = ae_logits_for_step(
        base_logits=base_logits,
        ae_model=ae_model,
        token_ids_model_device=token_ids_for_logits,
        ae_device=ae_device,
        mode=mode,
        ae_logit_scale=args.ae_logit_scale,
        ae_logit_bias=args.ae_logit_bias,
    )

    if mode in {"ae_latent", "ae_recon"}:
        next_k_idx = choose_index_from_logits(
            logits_k,
            decode=args.decode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return int(token_ids[next_k_idx]), {
            "selection_space": "token_list",
            "selected_tokenlist_index": int(next_k_idx),
        }

    if mode in {"ae_latent_patch", "ae_recon_patch"}:
        patched = base_logits.clone()
        if args.outside_token_bias != 0.0:
            patched = patched + float(args.outside_token_bias)
            patched.index_add_(
                0,
                token_ids_for_logits,
                torch.full_like(logits_k, -float(args.outside_token_bias), device=patched.device),
            )
        patched.index_copy_(0, token_ids_for_logits, logits_k.to(patched.device))
        next_id = choose_index_from_logits(
            patched,
            decode=args.decode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        selected_k_idx = None
        if int(next_id) in set(token_ids):
            # This is only used for diagnostics, so a Python lookup is fine.
            selected_k_idx = int(token_ids.index(int(next_id)))
        return int(next_id), {
            "selection_space": "full_vocab_patched",
            "selected_tokenlist_index": selected_k_idx,
        }

    raise ValueError(f"Unsupported mode: {mode}")


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    mode: str,
    args: argparse.Namespace,
    token_ids: Sequence[int],
    token_ids_model_device: Optional[torch.Tensor],
    ae_model: Optional[GSMTokenListAE],
    ae_device: Optional[torch.device],
) -> Dict:
    input_device = model_input_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(input_device)

    stop_ids = eos_token_ids(tokenizer, model)
    generated_ids: List[int] = []
    tokenlist_selected = 0
    past_key_values = None
    next_input_ids = input_ids

    for _step in range(args.max_new_tokens):
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        base_logits = outputs.logits[0, -1, :]
        next_token_id, select_info = select_next_token_id(
            base_logits=base_logits,
            mode=mode,
            args=args,
            token_ids=token_ids,
            token_ids_model_device=token_ids_model_device,
            ae_model=ae_model,
            ae_device=ae_device,
        )

        generated_ids.append(int(next_token_id))
        if select_info.get("selected_tokenlist_index") is not None:
            tokenlist_selected += 1

        past_key_values = outputs.past_key_values
        if int(next_token_id) in stop_ids:
            break

        if args.stop_after_final_answer:
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if extract_final_answer(partial_text) is not None:
                break

        next_input_ids = torch.tensor([[int(next_token_id)]], dtype=torch.long, device=input_device)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=input_device),
            ],
            dim=1,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "num_generated_tokens": len(generated_ids),
        "tokenlist_selected_steps": int(tokenlist_selected),
    }


def empty_totals() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "num_correct": 0.0,
        "num_parse_failures": 0.0,
        "num_generated_tokens": 0.0,
        "num_tokenlist_selected_steps": 0.0,
    }


def finalize_metrics(
    totals: Dict[str, float],
    mode: str,
    args: argparse.Namespace,
    k: int,
) -> Dict:
    n = max(int(totals["num_samples"]), 1)
    return {
        "mode": mode,
        "input_meta": args.input_meta,
        "model_path": args.model_path,
        "token_list_json": args.token_list_json if mode_needs_ae(mode) else None,
        "checkpoint": args.checkpoint if mode_needs_ae(mode) else None,
        "k": int(k) if mode_needs_ae(mode) else None,
        "decode": args.decode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": int(totals["num_samples"]),
        "num_correct": int(totals["num_correct"]),
        "accuracy": float(totals["num_correct"]) / float(n),
        "num_parse_failures": int(totals["num_parse_failures"]),
        "parse_failure_rate": float(totals["num_parse_failures"]) / float(n),
        "avg_generated_tokens": float(totals["num_generated_tokens"]) / float(n),
        "avg_tokenlist_selected_steps": float(totals["num_tokenlist_selected_steps"]) / float(n),
        "ae_logit_scale": args.ae_logit_scale if mode_needs_ae(mode) else None,
        "ae_logit_bias": args.ae_logit_bias if mode_needs_ae(mode) else None,
        "outside_token_bias": args.outside_token_bias if mode.endswith("_patch") else None,
    }


def run_mode(
    mode: str,
    records: Sequence[Dict],
    model,
    tokenizer,
    args: argparse.Namespace,
    token_ids: Sequence[int],
    token_ids_model_device: Optional[torch.Tensor],
    ae_model: Optional[GSMTokenListAE],
    ae_device: Optional[torch.device],
    k: int,
) -> Dict:
    mode_dir = Path(args.output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = mode_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    totals = empty_totals()
    for idx, record in enumerate(records, start=1):
        sample_id = get_sample_id(record, idx - 1)
        prompt_text = build_prompt(record)
        gold_raw = record_gold_text(record)
        gold_answer = extract_final_answer(gold_raw)

        generation = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            mode=mode,
            args=args,
            token_ids=token_ids,
            token_ids_model_device=token_ids_model_device,
            ae_model=ae_model,
            ae_device=ae_device,
        )
        pred_answer = extract_final_answer(generation["generated_text"])
        is_correct = answers_match(pred_answer, gold_answer)

        totals["num_samples"] += 1
        totals["num_correct"] += 1 if is_correct else 0
        totals["num_parse_failures"] += 1 if pred_answer is None else 0
        totals["num_generated_tokens"] += int(generation["num_generated_tokens"])
        totals["num_tokenlist_selected_steps"] += int(generation["tokenlist_selected_steps"])

        append_jsonl(
            predictions_path,
            {
                "sample_id": sample_id,
                "mode": mode,
                "question": record.get("question"),
                "gold_answer_raw": gold_raw,
                "gold_answer": normalize_answer(gold_answer),
                "pred_answer_raw": pred_answer,
                "pred_answer": normalize_answer(pred_answer),
                "correct": bool(is_correct),
                "num_generated_tokens": int(generation["num_generated_tokens"]),
                "tokenlist_selected_steps": int(generation["tokenlist_selected_steps"]),
                "prompt_text": prompt_text,
                "generated_text": generation["generated_text"],
            },
        )

        if args.log_every > 0 and (idx % args.log_every == 0 or idx == len(records)):
            partial = finalize_metrics(totals, mode=mode, args=args, k=k)
            LOGGER.info(
                "%s %d/%d | acc=%.4f | parse_fail=%.4f | avg_tokens=%.1f",
                mode,
                idx,
                len(records),
                partial["accuracy"],
                partial["parse_failure_rate"],
                partial["avg_generated_tokens"],
            )

    metrics = finalize_metrics(totals, mode=mode, args=args, k=k)
    write_json(mode_dir / "metrics.json", metrics)
    LOGGER.info("Wrote %s", mode_dir / "metrics.json")
    LOGGER.info("Wrote %s", predictions_path)
    return metrics


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(args.input_meta), max_samples=args.max_samples)
    if not records:
        raise ValueError(f"No records found in {args.input_meta}")

    LOGGER.info("Loading tokenizer and model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = None if str(args.device_map).lower() in {"none", "null"} else args.device_map
    model_kwargs = {
        "torch_dtype": resolve_dtype(args.load_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    if device_map is None:
        model.to(resolve_device(args.ae_device))

    token_ids: List[int] = []
    token_ids_model_device: Optional[torch.Tensor] = None
    ae_model: Optional[GSMTokenListAE] = None
    ae_device: Optional[torch.device] = None
    k = 0

    if any(mode_needs_ae(mode) for mode in args.modes):
        token_list = load_token_list(args.token_list_json)
        token_ids = [int(token_id) for token_id in token_list["token_ids"]]
        k = len(token_ids)
        ae_device = resolve_device(args.ae_device)
        ae_model, checkpoint = load_ae_model(args.checkpoint, k=k, device=ae_device)
        token_ids_model_device = torch.tensor(token_ids, dtype=torch.long, device=model_input_device(model))
        LOGGER.info(
            "Loaded token-list AE | K=%d | checkpoint_epoch=%s | ae_device=%s",
            k,
            checkpoint.get("epoch"),
            ae_device,
        )

    summary: Dict[str, Dict] = {}
    for mode in args.modes:
        LOGGER.info("Running mode=%s on %d records", mode, len(records))
        summary[mode] = run_mode(
            mode=mode,
            records=records,
            model=model,
            tokenizer=tokenizer,
            args=args,
            token_ids=token_ids,
            token_ids_model_device=token_ids_model_device,
            ae_model=ae_model,
            ae_device=ae_device,
            k=k,
        )

    summary_path = output_dir / "summary_metrics.json"
    write_json(
        summary_path,
        {
            "modes": list(args.modes),
            "num_records": len(records),
            "metrics": summary,
        },
    )
    LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
