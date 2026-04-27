import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


LOGGER = logging.getLogger("build_gsm_token_list")


TEXT_FIELDS = [
    "question",
    "answer",
    "answer_text",
    "rationale",
    "final_answer",
    "prompt",
    "prompt_text",
    "messages",
    "text",
    "input",
    "output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GSM8K-specific token list for token-aligned AE."
    )
    parser.add_argument(
        "--build-mode",
        choices=["all_text", "true_only"],
        default="all_text",
        help="all_text is the active mode. true_only builds from train answer_token_ids only.",
    )
    parser.add_argument("--train-meta-jsonl", default=None)
    parser.add_argument("--val-meta-jsonl", default=None)
    parser.add_argument("--test-meta-jsonl", default=None)
    parser.add_argument("--train-logits-dir", required=True)
    parser.add_argument("--val-logits-dir", default=None)
    parser.add_argument("--test-logits-dir", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--teacher-topk",
        type=int,
        default=50,
        help="Deprecated compatibility arg; ignored by all_text and true_only.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=None,
        help="Deprecated compatibility arg; token list is not truncated in all_text.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_pt_files(logits_dir: Optional[str]) -> List[Path]:
    if logits_dir is None:
        return []
    root = Path(logits_dir)
    if not root.exists():
        LOGGER.warning("Logits dir does not exist: %s", root)
        return []
    return sorted(root.rglob("*.pt"))


def read_jsonl(path: Path) -> List[Dict]:
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
    return records


def stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "content" in value:
            return stringify_value(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(text for text in (stringify_value(item) for item in value) if text)
    return str(value)


def messages_to_text(messages) -> str:
    if not isinstance(messages, list):
        return stringify_value(messages)
    parts: List[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = stringify_value(message.get("content", ""))
            if content:
                parts.append(content)
        else:
            content = stringify_value(message)
            if content:
                parts.append(content)
    return "\n".join(parts)


def first_present_text(record: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        if key in record:
            text = stringify_value(record[key]).strip()
            if text:
                return text
    return ""


def build_full_text(record: Dict) -> str:
    prompt_text = first_present_text(record, ["prompt_text", "prompt"])
    answer_text = first_present_text(record, ["answer_text", "answer", "output"])
    if prompt_text and answer_text:
        return prompt_text + answer_text

    question = first_present_text(record, ["question", "input"])
    rationale = first_present_text(record, ["rationale"])
    final_answer = first_present_text(record, ["final_answer"])
    if question and (answer_text or rationale or final_answer):
        parts = [f"Question:\n{question}", "Answer:"]
        if rationale:
            parts.append(rationale)
        if answer_text:
            parts.append(answer_text)
        if final_answer and final_answer not in parts:
            parts.append(final_answer)
        return "\n\n".join(parts)

    if "messages" in record:
        message_text = messages_to_text(record["messages"]).strip()
        if message_text:
            return message_text

    parts = []
    seen = set()
    for key in TEXT_FIELDS:
        if key not in record:
            continue
        text = messages_to_text(record[key]) if key == "messages" else stringify_value(record[key])
        text = text.strip()
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    return "\n\n".join(parts)


def load_tokenizer(tokenizer_path: Optional[str], required: bool):
    if tokenizer_path is None:
        if required:
            raise ValueError("all_text mode requires --tokenizer-path to tokenize train meta text.")
        return None

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        LOGGER.info("Loaded tokenizer from %s", tokenizer_path)
        return tokenizer
    except Exception as exc:
        if required:
            raise RuntimeError(f"all_text mode requires a working tokenizer: {tokenizer_path}") from exc
        LOGGER.warning("Could not load tokenizer from %s: %s", tokenizer_path, exc)
        return None


def tokenize_text(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def collect_text_token_freq(
    meta_jsonl: str,
    tokenizer,
    log_every: int,
    split_name: str,
) -> Tuple[Counter, int, int]:
    path = Path(meta_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} meta jsonl not found: {path}")

    records = read_jsonl(path)
    token_freq: Counter = Counter()
    total_tokens = 0
    for idx, record in enumerate(records, start=1):
        full_text = build_full_text(record)
        token_ids = tokenize_text(tokenizer, full_text) if full_text else []
        token_freq.update(token_ids)
        total_tokens += len(token_ids)
        if log_every > 0 and (idx % log_every == 0 or idx == len(records)):
            LOGGER.info(
                "%s text scan %d/%d | text_tokens=%d | unique_text_tokens=%d",
                split_name,
                idx,
                len(records),
                total_tokens,
                len(token_freq),
            )

    return token_freq, len(records), total_tokens


def get_answer_token_ids(obj: Dict, path: Path) -> torch.Tensor:
    if "answer_token_ids" in obj:
        ids = obj["answer_token_ids"]
    elif "answer_ids" in obj:
        ids = obj["answer_ids"]
    else:
        raise KeyError(f"{path} is missing answer_token_ids/answer_ids")
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
    return ids.long().view(-1)


def collect_answer_token_freq(
    logits_dir: Optional[str],
    log_every: int,
    split_name: str,
) -> Tuple[Counter, int, int]:
    files = iter_pt_files(logits_dir)
    answer_freq: Counter = Counter()
    total_positions = 0
    for file_idx, path in enumerate(files, start=1):
        obj = load_pt(path)
        answer_ids = get_answer_token_ids(obj, path)
        answer_freq.update(int(token_id) for token_id in answer_ids.tolist())
        total_positions += int(answer_ids.numel())
        if log_every > 0 and (file_idx % log_every == 0 or file_idx == len(files)):
            LOGGER.info(
                "%s answer scan %d/%d | positions=%d | unique_answer_tokens=%d",
                split_name,
                file_idx,
                len(files),
                total_positions,
                len(answer_freq),
            )
        del obj, answer_ids
    return answer_freq, total_positions, len(files)


def sorted_by_text_frequency(token_ids: Sequence[int], token_freq: Counter) -> List[int]:
    return sorted((int(token_id) for token_id in token_ids), key=lambda t: (-token_freq.get(t, 0), t))


def sorted_by_answer_frequency(answer_freq: Counter) -> List[int]:
    return sorted((int(token_id) for token_id in answer_freq.keys()), key=lambda t: (-answer_freq[t], t))


def decode_token_texts(tokenizer, token_ids: Sequence[int]) -> List[str]:
    if tokenizer is None:
        return ["" for _ in token_ids]

    texts: List[str] = []
    for token_id in token_ids:
        try:
            text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        except TypeError:
            text = tokenizer.decode([int(token_id)])
        except Exception as exc:
            LOGGER.warning("Could not decode token id %s: %s", token_id, exc)
            text = ""
        texts.append(text)
    return texts


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_text_coverage(
    meta_jsonl: Optional[str],
    tokenizer,
    token_set: set[int],
    split_name: str,
    log_every: int,
) -> Dict:
    empty = {
        "meta_jsonl": meta_jsonl,
        "total_text_tokens": 0,
        "covered_text_tokens": 0,
        "oov_text_tokens": 0,
        "text_token_coverage": 0.0,
        "text_oov_rate": 0.0,
    }
    if meta_jsonl is None:
        return empty
    if tokenizer is None:
        LOGGER.warning("Skipping %s text coverage because tokenizer is unavailable.", split_name)
        return empty

    records = read_jsonl(Path(meta_jsonl))
    total = 0
    covered = 0
    for idx, record in enumerate(records, start=1):
        full_text = build_full_text(record)
        token_ids = tokenize_text(tokenizer, full_text) if full_text else []
        total += len(token_ids)
        covered += sum(1 for token_id in token_ids if int(token_id) in token_set)
        if log_every > 0 and (idx % log_every == 0 or idx == len(records)):
            LOGGER.info(
                "%s text coverage %d/%d | total=%d | covered=%d",
                split_name,
                idx,
                len(records),
                total,
                covered,
            )

    oov = total - covered
    return {
        "meta_jsonl": meta_jsonl,
        "total_text_tokens": int(total),
        "covered_text_tokens": int(covered),
        "oov_text_tokens": int(oov),
        "text_token_coverage": safe_div(covered, total),
        "text_oov_rate": safe_div(oov, total),
    }


def compute_answer_coverage(
    logits_dir: Optional[str],
    token_set: set[int],
    split_name: str,
    log_every: int,
) -> Dict:
    answer_freq, total_positions, num_files = collect_answer_token_freq(
        logits_dir=logits_dir,
        log_every=log_every,
        split_name=f"{split_name} coverage",
    )
    covered = sum(freq for token_id, freq in answer_freq.items() if int(token_id) in token_set)
    oov = total_positions - covered
    return {
        "logits_dir": logits_dir,
        "num_files": int(num_files),
        "total_answer_positions": int(total_positions),
        "covered_answer_positions": int(covered),
        "oov_answer_positions": int(oov),
        "answer_token_coverage": safe_div(covered, total_positions),
        "answer_oov_rate": safe_div(oov, total_positions),
    }


def compute_split_coverage(
    split_name: str,
    meta_jsonl: Optional[str],
    logits_dir: Optional[str],
    tokenizer,
    token_set: set[int],
    log_every: int,
) -> Dict:
    return {
        "full_text": compute_text_coverage(
            meta_jsonl=meta_jsonl,
            tokenizer=tokenizer,
            token_set=token_set,
            split_name=split_name,
            log_every=log_every,
        ),
        "answer_targets": compute_answer_coverage(
            logits_dir=logits_dir,
            token_set=token_set,
            split_name=split_name,
            log_every=log_every,
        ),
    }


def counter_to_json(counter: Counter) -> Dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(counter.items(), key=lambda item: int(item[0]))}


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_all_text_token_list(args: argparse.Namespace, tokenizer) -> Tuple[List[int], Counter, Counter, Dict]:
    text_freq, num_text_samples, num_text_tokens = collect_text_token_freq(
        meta_jsonl=args.train_meta_jsonl,
        tokenizer=tokenizer,
        log_every=args.log_every,
        split_name="train",
    )
    answer_freq, num_answer_positions, num_train_logit_files = collect_answer_token_freq(
        logits_dir=args.train_logits_dir,
        log_every=args.log_every,
        split_name="train",
    )
    text_token_set = set(int(token_id) for token_id in text_freq.keys())
    answer_token_set = set(int(token_id) for token_id in answer_freq.keys())
    forced_answer_tokens = answer_token_set - text_token_set
    token_ids = sorted_by_text_frequency(text_token_set | answer_token_set, text_freq)

    meta = {
        "build_mode": "all_text",
        "final_k": int(len(token_ids)),
        "num_train_text_samples": int(num_text_samples),
        "num_train_text_tokens": int(num_text_tokens),
        "num_unique_text_tokens": int(len(text_freq)),
        "num_unique_answer_tokens": int(len(answer_freq)),
        "num_forced_answer_tokens_added": int(len(forced_answer_tokens)),
        "num_train_answer_positions": int(num_answer_positions),
        "num_train_logit_files": int(num_train_logit_files),
        "source": "unique tokenizer ids from full GSM8K train text plus train answer_token_ids",
        "train_meta_jsonl": args.train_meta_jsonl,
        "train_logits_dir": args.train_logits_dir,
        "val_meta_jsonl": args.val_meta_jsonl,
        "val_logits_dir": args.val_logits_dir,
        "test_meta_jsonl": args.test_meta_jsonl,
        "test_logits_dir": args.test_logits_dir,
    }
    return token_ids, text_freq, answer_freq, meta


def build_true_only_token_list(args: argparse.Namespace) -> Tuple[List[int], Counter, Counter, Dict]:
    answer_freq, num_answer_positions, num_train_logit_files = collect_answer_token_freq(
        logits_dir=args.train_logits_dir,
        log_every=args.log_every,
        split_name="train",
    )
    if not answer_freq:
        raise ValueError(f"No answer_token_ids found under train logits dir: {args.train_logits_dir}")
    token_ids = sorted_by_answer_frequency(answer_freq)
    text_freq: Counter = Counter()
    meta = {
        "build_mode": "true_only",
        "final_k": int(len(token_ids)),
        "num_train_text_samples": 0,
        "num_train_text_tokens": 0,
        "num_unique_text_tokens": 0,
        "num_unique_answer_tokens": int(len(answer_freq)),
        "num_forced_answer_tokens_added": int(len(answer_freq)),
        "num_train_answer_positions": int(num_answer_positions),
        "num_train_logit_files": int(num_train_logit_files),
        "source": "fallback answer-token-only token list from train answer_token_ids",
        "train_meta_jsonl": args.train_meta_jsonl,
        "train_logits_dir": args.train_logits_dir,
        "val_meta_jsonl": args.val_meta_jsonl,
        "val_logits_dir": args.val_logits_dir,
        "test_meta_jsonl": args.test_meta_jsonl,
        "test_logits_dir": args.test_logits_dir,
    }
    return token_ids, text_freq, answer_freq, meta


def main() -> None:
    setup_logging()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_build_mode = args.build_mode
    if args.build_mode == "all_text" and args.train_meta_jsonl is None:
        LOGGER.warning("train-meta-jsonl not provided; fallback to answer-token-only token list.")
        effective_build_mode = "true_only"

    tokenizer = load_tokenizer(
        args.tokenizer_path,
        required=effective_build_mode == "all_text",
    )

    if effective_build_mode == "all_text":
        token_ids, token_freq, answer_token_freq, meta = build_all_text_token_list(args, tokenizer)
    else:
        token_ids, token_freq, answer_token_freq, meta = build_true_only_token_list(args)

    token_to_index = {str(int(token_id)): int(idx) for idx, token_id in enumerate(token_ids)}
    token_texts = decode_token_texts(tokenizer, token_ids)

    token_list = {
        "token_ids": [int(token_id) for token_id in token_ids],
        "token_texts": token_texts,
        "token_to_index": token_to_index,
        "meta": meta,
        "stats": {
            "token_freq": counter_to_json(token_freq),
            "answer_token_freq": counter_to_json(answer_token_freq),
        },
    }

    token_list_path = output_dir / "gsm_token_list.json"
    write_json(token_list_path, token_list)
    LOGGER.info("Wrote %s | build_mode=%s | final_k=%d", token_list_path, meta["build_mode"], len(token_ids))

    token_set = set(int(token_id) for token_id in token_ids)
    coverage_report = {
        "token_list_json": str(token_list_path),
        "meta": {
            "build_mode": meta["build_mode"],
            "final_k": int(len(token_ids)),
            "coverage_definition": "text coverage uses tokenizer ids from meta full_text; answer coverage uses answer_token_ids from logits .pt files.",
        },
        "splits": {
            "train": compute_split_coverage(
                split_name="train",
                meta_jsonl=args.train_meta_jsonl,
                logits_dir=args.train_logits_dir,
                tokenizer=tokenizer,
                token_set=token_set,
                log_every=args.log_every,
            )
        },
    }
    if args.val_meta_jsonl is not None or args.val_logits_dir is not None:
        coverage_report["splits"]["val"] = compute_split_coverage(
            split_name="val",
            meta_jsonl=args.val_meta_jsonl,
            logits_dir=args.val_logits_dir,
            tokenizer=tokenizer,
            token_set=token_set,
            log_every=args.log_every,
        )
    if args.test_meta_jsonl is not None or args.test_logits_dir is not None:
        coverage_report["splits"]["test"] = compute_split_coverage(
            split_name="test",
            meta_jsonl=args.test_meta_jsonl,
            logits_dir=args.test_logits_dir,
            tokenizer=tokenizer,
            token_set=token_set,
            log_every=args.log_every,
        )

    coverage_path = output_dir / "token_list_coverage_report.json"
    write_json(coverage_path, coverage_report)
    LOGGER.info("Wrote %s", coverage_path)


if __name__ == "__main__":
    main()
