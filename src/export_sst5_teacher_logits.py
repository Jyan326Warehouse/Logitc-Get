from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sst5_common import (
    build_content_token_space,
    build_sst5_cot_prompt,
    choose_candidate_tokens,
    extract_final_label_from_text,
    infer_label_space,
    project_path,
    read_jsonl,
    resolve_dtype,
    safe_name,
    split_path,
    validate_records_against_label_space,
    write_json,
)


LOGGER = logging.getLogger("export_sst5_teacher_logits")
CACHE_FORMAT = "token_sequence_k_space_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SST-5 teacher sequence cache in frozen token K-space. "
            "K is built before training from train text/labels/label_text and teacher CoT targets."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="data/sst5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--expected-num-labels", type=int, default=5)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for teacher CoT generation.")
    parser.add_argument(
        "--prefix-batch-size",
        type=int,
        default=4,
        help="Batch size for teacher-forced sequence logits extraction.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--load-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--generate-cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must stay enabled: teacher CoT tokens are part of the K-space build.",
    )
    parser.add_argument("--cot-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=None,
        help="Optional cap on generated teacher target tokens saved per sample.",
    )
    parser.add_argument(
        "--cot-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cot-temperature", type=float, default=0.7)
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


def default_output_dir(model_path: str) -> Path:
    return project_path("data/sst5/teacher_logits") / safe_name(model_path)


def load_teacher(model_path: str, args: argparse.Namespace, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad_token or eos_token; set one before export.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_load_dtype(args.load_dtype, device),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def batch_iter(records: Sequence[Dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


def _valid_prompt_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[int]:
    return input_ids[attention_mask.bool()].detach().cpu().to(dtype=torch.long).tolist()


def generate_teacher_sequences(
    split: str,
    records: List[Dict],
    tokenizer,
    model,
    label_space,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict]:
    if not args.generate_cot:
        raise ValueError("K-space sequence export requires --generate-cot; teacher CoT builds K.")
    max_sequence_tokens = (
        int(args.max_sequence_tokens)
        if args.max_sequence_tokens is not None
        else int(args.cot_max_new_tokens)
    )
    sequences: List[Dict] = []
    progress = tqdm(
        batch_iter(records, args.batch_size),
        total=(len(records) + args.batch_size - 1) // args.batch_size,
        desc=f"generate {split} teacher targets",
    )
    for start, batch in progress:
        prompts = [build_sst5_cot_prompt(str(record["text"]), label_space=label_space) for record in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        generate_kwargs = {
            "max_new_tokens": int(args.cot_max_new_tokens),
            "do_sample": bool(args.cot_do_sample),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.cot_do_sample:
            generate_kwargs["temperature"] = float(args.cot_temperature)
        with torch.no_grad():
            generated = model.generate(**encoded, **generate_kwargs)

        encoded_width = int(encoded["input_ids"].shape[1])
        for offset, record in enumerate(batch):
            prompt_ids = _valid_prompt_ids(
                encoded["input_ids"][offset],
                encoded["attention_mask"][offset],
            )
            if tokenizer.padding_side == "left":
                new_ids_tensor = generated[offset, encoded_width:]
            else:
                prompt_len = int(encoded["attention_mask"][offset].sum().item())
                new_ids_tensor = generated[offset, prompt_len:]
            target_ids = new_ids_tensor.detach().cpu().to(dtype=torch.long).tolist()
            target_ids = [int(token_id) for token_id in target_ids[:max_sequence_tokens]]
            if not target_ids:
                raise ValueError(
                    f"Teacher generated no target tokens for split={split} index={start + offset}"
                )
            decoded = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
            sequences.append(
                {
                    "index": start + offset,
                    "prompt": prompts[offset],
                    "prompt_ids": [int(v) for v in prompt_ids],
                    "target_ids": target_ids,
                    "teacher_cot_output": f"Reasoning: {decoded}",
                    "teacher_cot_pred": extract_final_label_from_text(decoded, label_space.label_values),
                    "text": str(record["text"]),
                    "label": int(record["label"]),
                    "label_text": str(record.get("label_text", "")),
                }
            )
    return sequences


def pad_teacher_forced_batch(
    sequences: Sequence[Dict],
    pad_token_id: int,
    max_length: int,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], List[List[int]]]:
    if not sequences:
        raise ValueError("Cannot pad an empty teacher-forced batch")
    full_rows: List[List[int]] = []
    target_positions: List[List[int]] = []
    for sequence in sequences:
        prompt_ids = [int(v) for v in sequence["prompt_ids"]]
        target_ids = [int(v) for v in sequence["target_ids"]]
        if not prompt_ids or not target_ids:
            raise ValueError(
                f"split sequence index={sequence.get('index')} has empty prompt or target"
            )
        full_ids = prompt_ids + target_ids
        if len(full_ids) > int(max_length):
            raise ValueError(
                f"full sequence length {len(full_ids)} exceeds max_length={max_length} for "
                f"index={sequence.get('index')}. Increase --max-length or lower prompt/target length."
            )
        # Full-sequence next-token training uses logits at positions 0..L-2 to
        # predict tokens 1..L-1 across prompt_ids + teacher target_ids.
        full_rows.append(full_ids)
        target_positions.append(list(range(len(full_ids) - 1)))

    max_len = max(len(row) for row in full_rows)
    input_ids = torch.full((len(full_rows), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(full_rows), max_len), dtype=torch.long)
    padded_positions: List[List[int]] = []
    for row_idx, row in enumerate(full_rows):
        length = len(row)
        left_pad = max_len - length
        input_ids[row_idx, left_pad:] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_idx, left_pad:] = 1
        padded_positions.append([left_pad + pos for pos in target_positions[row_idx]])
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, padded_positions


def gather_teacher_forced_logits_K(
    model,
    sequences: Sequence[Dict],
    candidate_token_ids: Sequence[int],
    pad_token_id: int,
    device: torch.device,
    max_length: int,
    save_dtype: torch.dtype,
) -> List[torch.Tensor]:
    encoded, target_positions = pad_teacher_forced_batch(
        sequences=sequences,
        pad_token_id=pad_token_id,
        max_length=max_length,
        device=device,
    )
    candidate_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        full_logits = model(**encoded).logits
    rows: List[torch.Tensor] = []
    for row_idx, positions in enumerate(target_positions):
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        logits_TV = full_logits[row_idx].index_select(dim=0, index=pos_tensor)
        logits_TK = logits_TV.index_select(dim=-1, index=candidate_ids)
        rows.append(logits_TK.detach().cpu().to(dtype=save_dtype))
    return rows


def full_sequence_source_and_target_ids(sequence: Dict) -> tuple[List[int], List[int]]:
    prompt_ids = [int(v) for v in sequence["prompt_ids"]]
    target_ids = [int(v) for v in sequence["target_ids"]]
    if not prompt_ids or not target_ids:
        raise ValueError(f"sequence index={sequence.get('index')} has empty prompt or target ids")
    full_ids = prompt_ids + target_ids
    return full_ids[:-1], full_ids[1:]


def map_token_ids_to_k_indices_and_mask(
    token_ids: Sequence[int],
    sequence: Dict,
    tokenizer,
    token_id_to_k_index: Dict[int, int],
    split: str,
    oov_path: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    target_k_indices: List[int] = []
    loss_mask: List[bool] = []
    oov_rows: List[Dict] = []
    for pos, token_id in enumerate(token_ids):
        if token_id not in token_id_to_k_index:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            oov_rows.append(
                {
                    "split": split,
                    "index": int(sequence["index"]),
                    "position": int(pos),
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "text": str(sequence.get("text", "")),
                    "label": int(sequence.get("label", -1)),
                    "label_text": str(sequence.get("label_text", "")),
                    "sequence_part": "prompt_ids+target_ids next-token target",
                    "policy": (
                        "raise"
                        if split == "train"
                        else "skip_ce_and_kl_for_this_position"
                    ),
                }
            )
            target_k_indices.append(-100)
            loss_mask.append(False)
            continue
        target_k_indices.append(int(token_id_to_k_index[token_id]))
        loss_mask.append(True)
    if oov_rows:
        if oov_path is not None:
            oov_path.parent.mkdir(parents=True, exist_ok=True)
            with oov_path.open("a", encoding="utf-8") as f:
                for row in oov_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if split == "train":
            first = oov_rows[0]
            raise ValueError(
                f"TRAIN split index={sequence['index']} has {len(oov_rows)} next-token "
                f"targets outside frozen K. This means K construction is incomplete. "
                f"First OOV token id={first['token_id']} text={first['token_text']!r}. "
                f"OOV records written to {oov_path}."
            )
    return (
        torch.tensor(target_k_indices, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.bool),
        len(oov_rows),
    )


def export_split(
    split: str,
    records: List[Dict],
    output_path: Path,
    metadata_path: Path,
    tokenizer,
    model,
    label_space,
    content_token_space,
    args: argparse.Namespace,
    device: torch.device,
    precomputed_sequences: Optional[List[Dict]] = None,
) -> Dict:
    if output_path.exists() and not args.overwrite:
        try:
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
        except TypeError:
            existing = torch.load(output_path, map_location="cpu")
        if existing.get("cache_format") != CACHE_FORMAT:
            raise ValueError(
                f"Existing cache {output_path} is not {CACHE_FORMAT}. "
                "Use --overwrite to replace the old cache."
            )
        LOGGER.info("Skipping existing cache: %s", output_path)
        return {"split": split, "path": str(output_path), "skipped_existing": True}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()
    oov_path = output_path.parent / f"{split}_oov_records.jsonl"
    if oov_path.exists():
        oov_path.unlink()

    sequences = precomputed_sequences
    if sequences is None:
        sequences = generate_teacher_sequences(
            split=split,
            records=records,
            tokenizer=tokenizer,
            model=model,
            label_space=label_space,
            args=args,
            device=device,
        )
    if len(sequences) != len(records):
        raise ValueError(f"Generated {len(sequences)} sequences for {len(records)} {split} records")

    save_dtype = resolve_dtype(args.save_dtype)
    token_id_to_k_index = {
        int(token_id): int(idx)
        for idx, token_id in enumerate(content_token_space.candidate_token_ids)
    }
    all_input_logitc_K: List[torch.Tensor] = []
    all_target_k_indices: List[torch.Tensor] = []
    all_teacher_next_token_ids: List[torch.Tensor] = []
    all_input_ids: List[torch.Tensor] = []
    all_loss_mask: List[torch.Tensor] = []
    labels: List[int] = []
    texts: List[str] = []
    label_texts: List[str] = []
    indices: List[int] = []
    teacher_cot_outputs: List[str] = []
    teacher_cot_preds: List[Optional[int]] = []
    total_oov_positions = 0

    sequence_batches = list(batch_iter(sequences, int(args.prefix_batch_size)))
    progress = tqdm(
        sequence_batches,
        total=len(sequence_batches),
        desc=f"export {split} K logits",
    )
    for _, sequence_batch in progress:
        batch_logits = gather_teacher_forced_logits_K(
            model=model,
            sequences=sequence_batch,
            candidate_token_ids=content_token_space.candidate_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            max_length=args.max_length,
            save_dtype=save_dtype,
        )
        if len(batch_logits) != len(sequence_batch):
            raise ValueError(
                f"Expected {len(sequence_batch)} K-logit rows, got {len(batch_logits)}"
            )
        for sequence, input_logitc_K in zip(sequence_batch, batch_logits):
            source_ids, next_token_ids = full_sequence_source_and_target_ids(sequence)
            target_k_indices, position_loss_mask, oov_count = map_token_ids_to_k_indices_and_mask(
                token_ids=next_token_ids,
                sequence=sequence,
                tokenizer=tokenizer,
                token_id_to_k_index=token_id_to_k_index,
                split=split,
                oov_path=oov_path,
            )
            total_oov_positions += int(oov_count)
            if input_logitc_K.ndim != 2 or input_logitc_K.shape != (
                len(next_token_ids),
                content_token_space.k,
            ):
                raise ValueError(
                    f"input_logitc_K for split={split} index={sequence['index']} must be "
                    f"[T,K]=[{len(next_token_ids)},{content_token_space.k}], "
                    f"got {list(input_logitc_K.shape)}"
                )
            all_input_logitc_K.append(input_logitc_K)
            all_target_k_indices.append(target_k_indices)
            all_teacher_next_token_ids.append(torch.tensor(next_token_ids, dtype=torch.long))
            all_input_ids.append(torch.tensor(source_ids, dtype=torch.long))
            all_loss_mask.append(position_loss_mask)
            labels.append(int(sequence["label"]))
            texts.append(str(sequence["text"]))
            label_texts.append(str(sequence["label_text"]))
            indices.append(int(sequence["index"]))
            teacher_cot_outputs.append(str(sequence["teacher_cot_output"]))
            teacher_cot_preds.append(sequence["teacher_cot_pred"])

            with metadata_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "index": int(sequence["index"]),
                            "text": str(sequence["text"]),
                            "label": int(sequence["label"]),
                            "label_text": str(sequence["label_text"]),
                            "sequence_length": int(target_k_indices.shape[0]),
                            "valid_loss_tokens": int(position_loss_mask.sum().item()),
                            "oov_positions": int(oov_count),
                            "prompt_token_count": int(len(sequence["prompt_ids"])),
                            "teacher_target_token_count": int(len(sequence["target_ids"])),
                            "teacher_cot_output": str(sequence["teacher_cot_output"]),
                            "teacher_cot_pred": sequence["teacher_cot_pred"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    cache = {
        "cache_format": CACHE_FORMAT,
        "source": "SetFit/sst5",
        "split": split,
        "model_path": args.model_path,
        "input_logitc_K": all_input_logitc_K,
        "target_k_indices": all_target_k_indices,
        "teacher_next_token_ids": all_teacher_next_token_ids,
        "input_ids": all_input_ids,
        "loss_mask": all_loss_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
        "label_texts": label_texts,
        "indices": indices,
        "teacher_cot_outputs": teacher_cot_outputs,
        "teacher_cot_preds": teacher_cot_preds,
        "label_values": [int(v) for v in content_token_space.label_values],
        "label_text_by_value": {
            str(k): v for k, v in sorted(content_token_space.label_text_by_value.items())
        },
        "candidate_texts": content_token_space.candidate_texts,
        "candidate_token_ids": [int(v) for v in content_token_space.candidate_token_ids],
        "token_counts": [int(v) for v in content_token_space.token_counts],
        "content_token_space": content_token_space.to_dict(),
        "k": int(content_token_space.k),
        "k_frozen": True,
        "prompt_template": (
            "SST-5 visible reasoning prompt. For each generated teacher token, "
            "teacher-forced next-token logits are gathered over frozen train-derived K "
            "for the full prompt_ids + target_ids sequence."
        ),
        "oov_records_path": str(oov_path),
        "cot_prompt_template": "Reasoning: <brief sentiment rationale> then Final label: <digit>",
        "max_sequence_tokens": (
            int(args.max_sequence_tokens)
            if args.max_sequence_tokens is not None
            else int(args.cot_max_new_tokens)
        ),
        "oov_policy": {
            "train": "raise_error_because_K_is_built_from_train",
            "validation_test": "record_oov_and_set_target_k_indices_to_-100_and_loss_mask_false",
        },
        "oov_positions": int(total_oov_positions),
    }
    torch.save(cache, output_path)
    lengths = [int(row.shape[0]) for row in all_target_k_indices]
    return {
        "split": split,
        "path": str(output_path),
        "metadata_path": str(metadata_path),
        "num_samples": len(all_target_k_indices),
        "num_tokens": int(sum(lengths)),
        "num_valid_loss_tokens": int(sum(int(mask.sum().item()) for mask in all_loss_mask)),
        "oov_positions": int(total_oov_positions),
        "oov_records_path": str(oov_path),
        "min_sequence_tokens": int(min(lengths)) if lengths else 0,
        "max_sequence_tokens": int(max(lengths)) if lengths else 0,
        "k": int(content_token_space.k),
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = project_path(args.output_dir) if args.output_dir else default_output_dir(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_space = infer_label_space(
        split_path(args.data_dir, "train"),
        expected_num_labels=args.expected_num_labels,
    )
    LOGGER.info(
        "Inferred label_values=%s num_labels=%d from train split",
        label_space.label_values,
        label_space.k,
    )

    tokenizer, model = load_teacher(args.model_path, args, device=device)
    label_candidate_spec = choose_candidate_tokens(tokenizer, label_space.label_values)
    LOGGER.info(
        "Using SST-5 label token variant=%s texts=%s ids=%s",
        label_candidate_spec.variant_name,
        label_candidate_spec.candidate_texts,
        label_candidate_spec.candidate_token_ids,
    )

    train_records = read_jsonl(split_path(args.data_dir, "train"), max_records=args.max_samples)
    validate_records_against_label_space(train_records, label_space, split="train")
    train_sequences = generate_teacher_sequences(
        split="train",
        records=train_records,
        tokenizer=tokenizer,
        model=model,
        label_space=label_space,
        args=args,
        device=device,
    )
    train_prompt_ids = [sequence["prompt_ids"] for sequence in train_sequences]
    train_target_ids = [sequence["target_ids"] for sequence in train_sequences]
    train_next_token_ids = [
        full_sequence_source_and_target_ids(sequence)[1]
        for sequence in train_sequences
    ]
    content_token_space = build_content_token_space(
        tokenizer=tokenizer,
        train_jsonl=split_path(args.data_dir, "train"),
        label_space=label_space,
        label_candidate_spec=label_candidate_spec,
        min_token_count=args.min_token_count,
        max_k=args.max_k,
        teacher_prompt_token_ids_by_record=train_prompt_ids,
        teacher_cot_token_ids_by_record=train_target_ids,
        teacher_next_token_ids_by_record=train_next_token_ids,
    )
    LOGGER.info(
        "Built frozen K-space | K=%d source_fields=%s teacher_tokens=%d",
        content_token_space.k,
        content_token_space.source_fields,
        content_token_space.teacher_next_token_count,
    )
    k_vocab_path = project_path("outputs/k_vocab/sst5_k_vocab.json")
    write_json(
        k_vocab_path,
        {
            "cache_format": CACHE_FORMAT,
            "source": "SetFit/sst5",
            "model_path": args.model_path,
            "data_dir": str(project_path(args.data_dir)),
            "k": int(content_token_space.k),
            "k_frozen": True,
            "candidate_token_ids": [int(v) for v in content_token_space.candidate_token_ids],
            "candidate_texts": [str(v) for v in content_token_space.candidate_texts],
            "token_counts": [int(v) for v in content_token_space.token_counts],
            "token_id_to_k_index": {
                str(int(token_id)): int(idx)
                for idx, token_id in enumerate(content_token_space.candidate_token_ids)
            },
            "content_token_space": content_token_space.to_dict(),
        },
    )
    LOGGER.info("Wrote K vocab: %s", k_vocab_path)

    split_summaries = []
    for split in args.splits:
        if split == "train":
            records = train_records
            precomputed = train_sequences
        else:
            records = read_jsonl(split_path(args.data_dir, split), max_records=args.max_samples)
            validate_records_against_label_space(records, label_space, split=split)
            precomputed = None
        summary = export_split(
            split=split,
            records=records,
            output_path=output_dir / f"{split}.pt",
            metadata_path=output_dir / f"{split}_records.jsonl",
            tokenizer=tokenizer,
            model=model,
            label_space=label_space,
            content_token_space=content_token_space,
            args=args,
            device=device,
            precomputed_sequences=precomputed,
        )
        split_summaries.append(summary)

    manifest = {
        "cache_format": CACHE_FORMAT,
        "source": "SetFit/sst5",
        "model_path": args.model_path,
        "data_dir": str(project_path(args.data_dir)),
        "output_dir": str(output_dir),
        "label_space": label_space.to_dict(),
        "label_token_spec": label_candidate_spec.to_dict(),
        "content_token_space": content_token_space.to_dict(),
        "k_vocab_path": str(k_vocab_path),
        "splits": split_summaries,
        "args": vars(args),
    }
    write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
