from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from sst5_common import (
    SST5LabelSpace,
    infer_label_space,
    load_pt,
    read_jsonl,
    split_path,
    validate_records_against_label_space,
)


class SST5JsonlDataset(Dataset):
    """Raw SST-5 JSONL dataset using text as input and label as metadata."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        label_space: Optional[SST5LabelSpace] = None,
        expected_num_labels: Optional[int] = 5,
        max_samples: Optional[int] = None,
    ) -> None:
        self.path = split_path(data_dir, split)
        self.split = split
        if label_space is None:
            label_space = infer_label_space(
                split_path(data_dir, "train"),
                expected_num_labels=expected_num_labels,
            )
        self.label_space = label_space
        self.records = read_jsonl(self.path, max_records=max_samples)
        validate_records_against_label_space(self.records, label_space, split=split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        return {
            "index": int(idx),
            "text": str(record["text"]),
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "label_text": str(record.get("label_text", "")),
        }


def _tensor_list(value, name: str, dtype: torch.dtype) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.ndim < 1:
            raise ValueError(f"{name} tensor must have at least one dimension")
        return [value[idx].to(dtype=dtype, copy=True).cpu() for idx in range(value.shape[0])]
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a tensor or a sequence of tensors/lists")
    rows: List[torch.Tensor] = []
    for idx, row in enumerate(value):
        tensor = row if torch.is_tensor(row) else torch.tensor(row)
        rows.append(tensor.to(dtype=dtype, copy=True).cpu())
        if rows[-1].ndim == 0:
            raise ValueError(f"{name}[{idx}] must not be scalar")
    return rows


class SST5TeacherLogitsDataset(Dataset):
    """
    Frozen K-space teacher cache for token-level SST-5 AE training.

    Required cache tensors/lists:
        input_logitc_K:     N items of [T,K]
        target_k_indices:   N items of [T], K-space teacher next-token targets
        teacher_next_token_ids: N items of [T], original tokenizer token ids

    The dataset never expands K. Any target outside [0,K) raises immediately.
    """

    def __init__(
        self,
        cache_path: str | Path,
        expected_num_labels: Optional[int] = 5,
        logits_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cache_path = Path(cache_path)
        obj = load_pt(self.cache_path)
        if "input_logitc_K" not in obj:
            if "teacher_logits" in obj:
                raise ValueError(
                    f"{self.cache_path} is an old [N,K] cache. Re-export a token sequence "
                    "cache with input_logitc_K [N,T,K] and target_k_indices [N,T]."
                )
            raise KeyError(f"Teacher cache {self.cache_path} missing key 'input_logitc_K'")
        for key in ("target_k_indices", "teacher_next_token_ids", "candidate_token_ids"):
            if key not in obj:
                raise KeyError(f"Teacher cache {self.cache_path} missing key {key!r}")

        self.input_logitc_K = _tensor_list(obj["input_logitc_K"], "input_logitc_K", logits_dtype)
        self.target_k_indices = _tensor_list(obj["target_k_indices"], "target_k_indices", torch.long)
        self.teacher_next_token_ids = _tensor_list(
            obj["teacher_next_token_ids"],
            "teacher_next_token_ids",
            torch.long,
        )
        input_ids_obj = obj.get("input_ids", obj["teacher_next_token_ids"])
        self.input_ids = _tensor_list(input_ids_obj, "input_ids", torch.long)
        loss_mask_obj = obj.get("loss_mask")
        self.loss_mask = (
            _tensor_list(loss_mask_obj, "loss_mask", torch.bool)
            if loss_mask_obj is not None
            else [row.ne(-100) for row in self.target_k_indices]
        )

        n = len(self.input_logitc_K)
        if not (
            len(self.target_k_indices)
            == len(self.teacher_next_token_ids)
            == len(self.input_ids)
            == len(self.loss_mask)
            == n
        ):
            raise ValueError(f"Cache sequence counts do not match in {self.cache_path}")
        if n == 0:
            raise ValueError(f"Cache has no samples: {self.cache_path}")

        self.k = int(obj.get("k", self.input_logitc_K[0].shape[-1]))
        self.candidate_token_ids = [int(v) for v in obj["candidate_token_ids"]]
        if len(self.candidate_token_ids) != self.k:
            raise ValueError(
                f"candidate_token_ids length {len(self.candidate_token_ids)} does not match K={self.k}"
            )
        self.token_id_to_k_index = {
            int(token_id): int(idx) for idx, token_id in enumerate(self.candidate_token_ids)
        }
        k_frozen = obj.get("k_frozen", obj.get("content_token_space", {}).get("frozen", True))
        if not bool(k_frozen):
            raise ValueError(f"K-space in {self.cache_path} is not marked frozen")

        for idx in range(n):
            logits = self.input_logitc_K[idx]
            target = self.target_k_indices[idx]
            next_ids = self.teacher_next_token_ids[idx]
            input_ids = self.input_ids[idx]
            mask = self.loss_mask[idx]
            if logits.ndim != 2 or logits.shape[-1] != self.k:
                raise ValueError(
                    f"input_logitc_K[{idx}] must have shape [T,{self.k}], got {list(logits.shape)}"
                )
            t = int(logits.shape[0])
            for name, tensor in (
                ("target_k_indices", target),
                ("teacher_next_token_ids", next_ids),
                ("input_ids", input_ids),
                ("loss_mask", mask),
            ):
                if tensor.ndim != 1 or int(tensor.shape[0]) != t:
                    raise ValueError(
                        f"{name}[{idx}] must have shape [T={t}], got {list(tensor.shape)}"
                    )
            valid = mask.bool() & target.ne(-100)
            if bool(valid.any().item()):
                valid_targets = target[valid]
                min_target = int(valid_targets.min().item())
                max_target = int(valid_targets.max().item())
                if min_target < 0 or max_target >= self.k:
                    raise ValueError(
                        f"target_k_indices[{idx}] contains target outside frozen K=[0,{self.k - 1}]: "
                        f"min={min_target} max={max_target}"
                    )

        self.label_values = [int(v) for v in obj.get("label_values", [])]
        if expected_num_labels is not None and self.label_values:
            if len(self.label_values) != int(expected_num_labels):
                raise ValueError(f"Expected {expected_num_labels} labels, got {len(self.label_values)}")
        self.num_labels = len(self.label_values)
        self.labels = torch.tensor(
            [int(v) for v in obj.get("labels", [-1] * n)],
            dtype=torch.long,
        )
        if int(self.labels.numel()) != n:
            raise ValueError(f"labels length does not match N={n}")

        self.texts = [str(v) for v in obj.get("texts", [""] * n)]
        self.label_texts = [str(v) for v in obj.get("label_texts", [""] * n)]
        self.source_indices = [int(v) for v in obj.get("indices", list(range(n)))]
        self.teacher_cot_outputs = [str(v) for v in obj.get("teacher_cot_outputs", [""] * n)]
        if not (
            len(self.texts)
            == len(self.label_texts)
            == len(self.source_indices)
            == len(self.teacher_cot_outputs)
            == n
        ):
            raise ValueError(f"Cache metadata lengths do not match N={n}: {self.cache_path}")

        self.sequence_lengths = [int(row.shape[0]) for row in self.target_k_indices]
        self.total_tokens = int(sum(self.sequence_lengths))
        self.metadata = {
            key: obj.get(key)
            for key in (
                "source",
                "split",
                "model_path",
                "label_values",
                "label_text_by_value",
                "candidate_texts",
                "candidate_token_ids",
                "content_token_space",
                "k",
                "k_frozen",
                "prompt_template",
                "cot_prompt_template",
                "cache_format",
                "max_sequence_tokens",
            )
        }

    def __len__(self) -> int:
        return len(self.input_logitc_K)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.loss_mask[idx].to(dtype=torch.long),
            "input_logitc_K": self.input_logitc_K[idx],
            "target_k_indices": self.target_k_indices[idx],
            "teacher_next_token_ids": self.teacher_next_token_ids[idx],
            "loss_mask": self.loss_mask[idx],
            "label": self.labels[idx],
            "text": self.texts[idx],
            "label_text": self.label_texts[idx],
            "index": torch.tensor(self.source_indices[idx], dtype=torch.long),
            "teacher_cot_output": self.teacher_cot_outputs[idx],
        }


def make_target_onehot_K(
    target_k_indices: torch.Tensor,
    loss_mask: torch.Tensor,
    k: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if target_k_indices.ndim != 2:
        raise ValueError(
            f"target_k_indices must have shape [B,T], got {list(target_k_indices.shape)}"
        )
    if loss_mask.shape != target_k_indices.shape:
        raise ValueError(
            f"loss_mask shape {list(loss_mask.shape)} must match target_k_indices "
            f"{list(target_k_indices.shape)}"
        )
    safe_target = target_k_indices.clamp_min(0)
    onehot = torch.nn.functional.one_hot(safe_target, num_classes=int(k)).to(dtype=dtype)
    valid = loss_mask.bool() & target_k_indices.ne(-100)
    onehot = onehot * valid.unsqueeze(-1).to(dtype=dtype)
    expected_shape = (*target_k_indices.shape, int(k))
    if tuple(onehot.shape) != expected_shape:
        raise ValueError(
            f"target_onehot_K must have shape {list(expected_shape)}, got {list(onehot.shape)}"
        )
    return onehot


def sst5_k_space_collate_fn(batch: Sequence[Dict], include_target_onehot: bool = False) -> Dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    k = int(batch[0]["input_logitc_K"].shape[-1])
    max_t = max(int(item["input_logitc_K"].shape[0]) for item in batch)
    logits_dtype = batch[0]["input_logitc_K"].dtype

    input_ids = torch.zeros(batch_size, max_t, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_t, dtype=torch.long)
    input_logitc_K = torch.zeros(batch_size, max_t, k, dtype=logits_dtype)
    target_k_indices = torch.full((batch_size, max_t), -100, dtype=torch.long)
    teacher_next_token_ids = torch.full((batch_size, max_t), -100, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)

    labels = []
    indices = []
    texts = []
    label_texts = []
    teacher_cot_outputs = []
    for row_idx, item in enumerate(batch):
        row_logits = item["input_logitc_K"]
        if row_logits.ndim != 2 or int(row_logits.shape[-1]) != k:
            raise ValueError(
                f"input_logitc_K item must have shape [T,{k}], got {list(row_logits.shape)}"
            )
        t = int(row_logits.shape[0])
        input_logitc_K[row_idx, :t] = row_logits
        input_ids[row_idx, :t] = item["input_ids"].to(dtype=torch.long)
        attention_mask[row_idx, :t] = item["attention_mask"].to(dtype=torch.long)
        target_k_indices[row_idx, :t] = item["target_k_indices"].to(dtype=torch.long)
        teacher_next_token_ids[row_idx, :t] = item["teacher_next_token_ids"].to(dtype=torch.long)
        loss_mask[row_idx, :t] = item["loss_mask"].to(dtype=torch.bool)
        labels.append(item["label"])
        indices.append(item["index"])
        texts.append(item["text"])
        label_texts.append(item["label_text"])
        teacher_cot_outputs.append(item["teacher_cot_output"])

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_logitc_K": input_logitc_K,
        "target_k_indices": target_k_indices,
        "teacher_next_token_ids": teacher_next_token_ids,
        "loss_mask": loss_mask,
        "label": torch.stack(labels).to(dtype=torch.long),
        "index": torch.stack(indices).to(dtype=torch.long),
        "text": texts,
        "label_text": label_texts,
        "teacher_cot_output": teacher_cot_outputs,
    }
    if include_target_onehot:
        out["target_onehot_K"] = make_target_onehot_K(
            target_k_indices=target_k_indices,
            loss_mask=loss_mask,
            k=k,
            dtype=logits_dtype,
        )
    return out
