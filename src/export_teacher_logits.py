import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_TEMPLATE = """Solve the following math problem step by step.

Question:
{question}

Answer:
"""

LIGHTWEIGHT_SPLIT_SAMPLES = {
    "train": 3200,
    "test": 800,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export teacher next-token logits for GSM8K answers. "
            "Outputs answer_logits [T, V] and answer_token_ids [T]."
        )
    )
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--dataset_name", "--dataset-name", dest="dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", "--dataset-config", dest="dataset_config", default="main")
    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Dataset split for dataset mode, or logical output split for --input-meta "
            "(for example ae_train, ae_val, main_test)."
        ),
    )
    parser.add_argument(
        "--input-meta",
        "--meta-file",
        dest="input_meta",
        default=None,
        help="Optional JSONL meta file with prompt_text and answer_text. Use this for ae_train/ae_val/main_test.",
    )
    parser.add_argument(
        "--output-split",
        default=None,
        help="Override output logits subdirectory name. Defaults to inferred split.",
    )
    parser.add_argument("--output_root", "--output-root", dest="output_root", default="data")
    parser.add_argument("--meta-out", default=None, help="Optional output meta JSONL path.")
    parser.add_argument(
        "--max_samples",
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Limit exported samples. Dataset mode defaults to train=3200/test=800 unless --full-split is set.",
    )
    parser.add_argument("--full_split", "--full-split", dest="full_split", action="store_true")
    parser.add_argument("--start_idx", "--start-idx", dest="start_idx", type=int, default=0)
    parser.add_argument(
        "--load_dtype",
        "--load-dtype",
        dest="load_dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--save_dtype",
        "--save-dtype",
        dest="save_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust_remote_code", "--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--skip_existing", "--skip-existing", dest="skip_existing", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def resolve_dataset_split(dataset_name: str, split: str) -> str:
    dataset_aliases = {
        "openai/gsm8k": {
            "validation": "test",
        }
    }
    return dataset_aliases.get(dataset_name, {}).get(split, split)


def infer_split_from_meta(input_meta: Optional[str], fallback_split: str) -> str:
    if input_meta is None:
        return fallback_split
    name = Path(input_meta).stem
    if name.startswith("hidden_") or name.startswith("logits_"):
        name = name.split("_", 1)[1]
    return name


def find_prompt_token_len(
    tokenizer,
    prompt_text: str,
    full_text: str,
    add_special_tokens: bool = True,
) -> Tuple[int, Dict[str, torch.Tensor]]:
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    full_kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": add_special_tokens,
    }
    if getattr(tokenizer, "is_fast", False):
        full_kwargs["return_offsets_mapping"] = True

    full_enc = tokenizer(full_text, **full_kwargs)
    prompt_ids = prompt_enc["input_ids"][0]
    full_ids = full_enc["input_ids"][0]

    if len(full_ids) >= len(prompt_ids) and torch.equal(full_ids[: len(prompt_ids)], prompt_ids):
        full_enc.pop("offset_mapping", None)
        return len(prompt_ids), full_enc

    offsets = full_enc.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Could not locate prompt/answer token boundary: prefix alignment failed "
            "and tokenizer offset_mapping is unavailable."
        )

    boundary = len(prompt_text)
    prompt_len = 0
    for start, end in offsets[0].tolist():
        if start == 0 and end == 0:
            prompt_len += 1
            continue
        if end <= boundary:
            prompt_len += 1
            continue
        break

    full_enc.pop("offset_mapping", None)
    return prompt_len, full_enc


def ensure_dirs(output_root: Path, output_split: str) -> Tuple[Path, Path]:
    meta_dir = output_root / "meta"
    logits_dir = output_root / "logits" / output_split
    meta_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir, logits_dir


def read_jsonl(path: Path, max_records: Optional[int] = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "answer_text" not in record:
                raise KeyError(f"Missing answer_text on line {line_idx}: {path}")
            if "prompt_text" not in record and "question" not in record:
                raise KeyError(f"Missing prompt_text/question on line {line_idx}: {path}")
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    return records


def load_dataset_records(args: argparse.Namespace, effective_max_samples: Optional[int]) -> List[Dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for dataset mode. Use --input-meta to avoid it.") from exc

    resolved_split = resolve_dataset_split(args.dataset_name, args.split)
    if resolved_split not in {"train", "test", "validation"}:
        raise ValueError(
            f"Split '{args.split}' is not a raw dataset split. "
            "Use --input-meta for logical splits such as ae_train, ae_val, or main_test."
        )

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=resolved_split)
    end_idx = (
        len(dataset)
        if effective_max_samples is None
        else min(len(dataset), args.start_idx + effective_max_samples)
    )
    records: List[Dict] = []
    for idx in range(args.start_idx, end_idx):
        ex = dataset[idx]
        sample_id = f"gsm8k_{args.dataset_config}_{resolved_split}_{idx:06d}"
        prompt_text = build_prompt(ex["question"])
        records.append(
            {
                "sample_id": sample_id,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": resolved_split,
                "index": idx,
                "question": ex["question"],
                "answer_text": ex["answer"],
                "prompt_text": prompt_text,
            }
        )
    print(f"Total dataset size = {len(dataset)}")
    print(f"Export range       = [{args.start_idx}, {end_idx})")
    return records


def prepare_records(args: argparse.Namespace, effective_max_samples: Optional[int]) -> List[Dict]:
    if args.input_meta is not None:
        meta_path = resolve_project_path(args.input_meta)
        records = read_jsonl(meta_path, max_records=None)
        start = min(args.start_idx, len(records))
        end = len(records) if effective_max_samples is None else min(len(records), start + effective_max_samples)
        print(f"Input meta         = {meta_path}")
        print(f"Meta records       = {len(records)}")
        print(f"Export range       = [{start}, {end})")
        return records[start:end]
    return load_dataset_records(args, effective_max_samples)


def record_prompt_and_answer(record: Dict) -> Tuple[str, str]:
    if "prompt_text" in record:
        prompt_text = record["prompt_text"]
    else:
        prompt_text = build_prompt(record["question"])
    return prompt_text, record["answer_text"]


def data_relative(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path.resolve())


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.full_split:
        effective_max_samples = None
    elif args.max_samples is not None:
        effective_max_samples = args.max_samples
    elif args.input_meta is not None:
        effective_max_samples = None
    else:
        resolved_raw_split = resolve_dataset_split(args.dataset_name, args.split)
        effective_max_samples = LIGHTWEIGHT_SPLIT_SAMPLES.get(resolved_raw_split)

    inferred_split = infer_split_from_meta(args.input_meta, resolve_dataset_split(args.dataset_name, args.split))
    output_split = args.output_split or inferred_split

    output_root = resolve_project_path(args.output_root)
    meta_dir, logits_dir = ensure_dirs(output_root, output_split)
    meta_path = resolve_project_path(args.meta_out) if args.meta_out else meta_dir / f"logits_{output_split}.jsonl"

    load_dtype = resolve_dtype(args.load_dtype)
    save_dtype = resolve_dtype(args.save_dtype)

    print("=" * 80)
    print("Loading tokenizer and model...")
    print(f"model_path   = {args.model_path}")
    print(f"output_split = {output_split}")
    print(f"max_samples  = {effective_max_samples}")
    print(f"output_root  = {output_root}")
    print(f"logits_dir   = {logits_dir}")
    print(f"meta_path    = {meta_path}")
    print(f"load_dtype   = {load_dtype}")
    print(f"save_dtype   = {save_dtype}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=load_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device

    records = prepare_records(args, effective_max_samples)
    if not records:
        raise ValueError("No records to export.")

    num_ok = 0
    num_skip = 0
    num_err = 0
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8", newline="\n") as f_meta:
        for local_idx, record in enumerate(tqdm(records, desc="Exporting teacher logits")):
            sample_id = record.get("sample_id") or f"gsm8k_{output_split}_{local_idx:06d}"
            logits_file = logits_dir / f"{sample_id}.pt"

            if args.skip_existing and logits_file.exists():
                num_skip += 1
                continue

            try:
                prompt_text, answer_text = record_prompt_and_answer(record)
                full_text = prompt_text + answer_text
                prompt_len, full_enc = find_prompt_token_len(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    full_text=full_text,
                    add_special_tokens=True,
                )

                input_ids = full_enc["input_ids"]
                attention_mask = full_enc["attention_mask"]
                full_len = int(input_ids.shape[1])
                answer_len = full_len - int(prompt_len)
                if answer_len <= 0:
                    raise ValueError(f"answer_len <= 0, sample_id={sample_id}")
                if prompt_len < 1:
                    raise ValueError(f"prompt_len < 1, sample_id={sample_id}")

                input_ids_gpu = input_ids.to(model_device)
                attention_mask_gpu = attention_mask.to(model_device)
                outputs = model(
                    input_ids=input_ids_gpu,
                    attention_mask=attention_mask_gpu,
                    use_cache=False,
                )
                logits = outputs.logits[0]

                answer_token_ids = input_ids[0, prompt_len:full_len].cpu()
                answer_logits = logits[prompt_len - 1 : full_len - 1].detach().to(save_dtype).cpu()
                if answer_logits.shape[0] != answer_token_ids.shape[0]:
                    raise ValueError(
                        f"length mismatch: answer_logits={answer_logits.shape[0]}, "
                        f"answer_token_ids={answer_token_ids.shape[0]}, sample_id={sample_id}"
                    )

                save_obj = {
                    "sample_id": sample_id,
                    "answer_logits": answer_logits,
                    "answer_token_ids": answer_token_ids,
                    "answer_ids": answer_token_ids,
                    "full_input_ids": input_ids[0].cpu(),
                    "prompt_len": int(prompt_len),
                    "full_len": int(full_len),
                    "answer_len": int(answer_len),
                    "source_record": record,
                }
                torch.save(save_obj, logits_file)

                meta_record = dict(record)
                meta_record.update(
                    {
                        "sample_id": sample_id,
                        "logits_path": data_relative(logits_file, output_root),
                        "logits_split": output_split,
                        "prompt_text": prompt_text,
                        "answer_text": answer_text,
                        "prompt_len": int(prompt_len),
                        "full_len": int(full_len),
                        "answer_len": int(answer_len),
                    }
                )
                f_meta.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
                num_ok += 1

                del outputs, logits, answer_logits, input_ids_gpu, attention_mask_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                err_record = {
                    "sample_id": sample_id,
                    "error": str(exc),
                }
                print(f"[ERROR] {json.dumps(err_record, ensure_ascii=False)}")
                num_err += 1

    print()
    print("=" * 80)
    print("Done.")
    print(f"OK    : {num_ok}")
    print(f"SKIP  : {num_skip}")
    print(f"ERROR : {num_err}")
    print(f"Logits dir written to: {logits_dir}")
    print(f"Meta file written to : {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
