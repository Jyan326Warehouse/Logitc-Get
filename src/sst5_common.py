from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABEL_CANDIDATE_PREFIX = " "


@dataclass(frozen=True)
class SST5LabelSpace:
    label_values: List[int]
    label_text_by_value: Dict[int, str]
    expected_num_labels: Optional[int] = 5
    num_train_records: int = 0
    label_counts: Dict[int, int] = field(default_factory=dict)
    text_stats: Dict[str, float] = field(default_factory=dict)
    train_text_label_sha256: str = ""

    @property
    def k(self) -> int:
        return len(self.label_values)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["label_text_by_value"] = {
            str(k): v for k, v in sorted(self.label_text_by_value.items())
        }
        out["label_counts"] = {str(k): v for k, v in sorted(self.label_counts.items())}
        return out


@dataclass(frozen=True)
class CandidateTokenSpec:
    candidate_texts: List[str]
    candidate_token_ids: List[int]
    prompt_trailing_space: bool
    variant_name: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SST5ContentTokenSpace:
    candidate_token_ids: List[int]
    candidate_texts: List[str]
    token_counts: List[int]
    label_values: List[int]
    label_token_ids: List[int]
    label_candidate_indices: List[int]
    text_candidate_indices: List[int]
    label_to_candidate_index: Dict[int, int]
    label_text_by_value: Dict[int, str]
    num_train_records: int
    source_fields: List[str]
    teacher_prompt_token_count: int = 0
    teacher_cot_token_count: int = 0
    teacher_next_token_count: int = 0
    frozen: bool = True
    min_token_count: int = 1
    max_k: Optional[int] = None
    train_text_label_sha256: str = ""

    @property
    def k(self) -> int:
        return len(self.candidate_token_ids)

    @property
    def num_labels(self) -> int:
        return len(self.label_values)

    @property
    def k_label(self) -> int:
        return len(self.label_candidate_indices)

    @property
    def k_text(self) -> int:
        return len(self.text_candidate_indices)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["label_to_candidate_index"] = {
            str(k): int(v) for k, v in sorted(self.label_to_candidate_index.items())
        }
        out["label_text_by_value"] = {
            str(k): v for k, v in sorted(self.label_text_by_value.items())
        }
        out["k"] = self.k
        out["k_label"] = self.k_label
        out["k_text"] = self.k_text
        out["frozen"] = bool(self.frozen)
        out["token_id_to_k_index"] = {
            str(int(token_id)): int(idx)
            for idx, token_id in enumerate(self.candidate_token_ids)
        }
        out["space_parts"] = {
            "label": {
                "size": self.k_label,
                "indices": [int(v) for v in self.label_candidate_indices],
            },
            "text": {
                "size": self.k_text,
                "indices": [int(v) for v in self.text_candidate_indices],
            },
        }
        return out


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_json(path: str | Path) -> Dict:
    with project_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Dict) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: str | Path, obj: Dict) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, max_records: Optional[int] = None) -> List[Dict]:
    path = project_path(path)
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for field in ("text", "label", "label_text"):
                if field not in obj:
                    raise KeyError(f"Missing '{field}' in {path} line {line_idx}")
            obj["label"] = int(obj["label"])
            records.append(obj)
            if max_records is not None and len(records) >= max_records:
                break
    return records


def resolve_sst5_jsonl_dir(data_dir: str | Path) -> Path:
    data_dir = project_path(data_dir)
    if (data_dir / "jsonl").is_dir():
        return data_dir / "jsonl"
    return data_dir


def split_path(data_dir: str | Path, split: str) -> Path:
    return resolve_sst5_jsonl_dir(data_dir) / f"{split}.jsonl"


def infer_label_space(
    train_jsonl: str | Path,
    expected_num_labels: Optional[int] = 5,
    require_contiguous_zero_based: bool = True,
) -> SST5LabelSpace:
    records = read_jsonl(train_jsonl)
    if not records:
        raise ValueError(f"Train split is empty: {project_path(train_jsonl)}")
    label_values = sorted({int(r["label"]) for r in records})
    if expected_num_labels is not None and len(label_values) != int(expected_num_labels):
        raise ValueError(
            f"Expected {expected_num_labels} labels, got {len(label_values)}: {label_values}"
        )

    if require_contiguous_zero_based:
        expected_values = list(range(len(label_values)))
        if label_values != expected_values:
            raise ValueError(
                "SST-5 AE uses original labels directly as CE targets, so labels must be "
                f"contiguous zero-based values. Got {label_values}, expected {expected_values}."
            )

    label_text_by_value: Dict[int, str] = {}
    label_counts: Dict[int, int] = {int(value): 0 for value in label_values}
    text_lengths: List[int] = []
    digest = hashlib.sha256()
    for record in records:
        label = int(record["label"])
        review_text = str(record.get("text", ""))
        if not review_text.strip():
            raise ValueError("SST-5 train record has empty text; cannot build task label space.")
        label_counts[label] = label_counts.get(label, 0) + 1
        text_lengths.append(len(review_text))
        digest.update(review_text.encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\n")
        text = str(record.get("label_text", ""))
        existing = label_text_by_value.get(label)
        if existing is None:
            label_text_by_value[label] = text
        elif existing != text:
            raise ValueError(
                f"Inconsistent label_text for label {label}: {existing!r} vs {text!r}"
            )

    return SST5LabelSpace(
        label_values=label_values,
        label_text_by_value=label_text_by_value,
        expected_num_labels=expected_num_labels,
        num_train_records=len(records),
        label_counts=label_counts,
        text_stats={
            "min_chars": float(min(text_lengths)),
            "max_chars": float(max(text_lengths)),
            "mean_chars": float(sum(text_lengths) / len(text_lengths)),
        },
        train_text_label_sha256=digest.hexdigest(),
    )


def validate_records_against_label_space(
    records: Sequence[Dict],
    label_space: SST5LabelSpace,
    split: str,
) -> None:
    allowed = set(int(v) for v in label_space.label_values)
    for idx, record in enumerate(records):
        label = int(record["label"])
        if label not in allowed:
            raise ValueError(
                f"Split {split} record {idx} has label {label}, not in {label_space.label_values}"
            )


def build_label_scale_text(label_space: SST5LabelSpace) -> str:
    lines = []
    for value in label_space.label_values:
        label_text = label_space.label_text_by_value.get(int(value), "")
        if label_text:
            lines.append(f"{value} = {label_text}")
        else:
            lines.append(str(value))
    return "\n".join(lines)


def build_sst5_label_prompt(
    review_text: str,
    label_space: SST5LabelSpace,
    prompt_trailing_space: bool = False,
) -> str:
    suffix = "Label:" + (" " if prompt_trailing_space else "")
    min_label = min(label_space.label_values)
    max_label = max(label_space.label_values)
    return (
        "You are an SST-5 sentiment classifier.\n"
        f"Read the movie-review text and output exactly one digit from {min_label} to {max_label}.\n"
        "Do not output words, punctuation, or explanation.\n\n"
        "Label scale:\n"
        f"{build_label_scale_text(label_space)}\n\n"
        "Text:\n"
        f"{review_text.strip()}\n\n"
        f"{suffix}"
    )


def build_sst5_prompt(
    review_text: str,
    label_space: SST5LabelSpace,
    prompt_trailing_space: bool = False,
) -> str:
    return build_sst5_label_prompt(
        review_text=review_text,
        label_space=label_space,
        prompt_trailing_space=prompt_trailing_space,
    )


def build_sst5_cot_prompt(review_text: str, label_space: SST5LabelSpace) -> str:
    min_label = min(label_space.label_values)
    max_label = max(label_space.label_values)
    return (
        "You are an SST-5 sentiment classifier.\n"
        "Read the movie-review text. First write a brief reasoning paragraph, "
        f"then write the final result as one digit from {min_label} to {max_label}.\n"
        "Use exactly this format:\n"
        "Reasoning: <brief sentiment rationale>\n"
        "Final label: <digit>\n\n"
        "Label scale:\n"
        f"{build_label_scale_text(label_space)}\n\n"
        "Text:\n"
        f"{review_text.strip()}\n\n"
        "Reasoning:"
    )


def build_sst5_cot_logits_prompt(review_text: str, label_space: SST5LabelSpace) -> str:
    return build_sst5_cot_prompt(review_text=review_text, label_space=label_space)


def extract_final_label_from_text(text: str, label_values: Sequence[int]) -> Optional[int]:
    allowed = {int(v) for v in label_values}
    patterns = [
        r"Final\s+label\s*:\s*([0-9]+)",
        r"Result\s*:\s*([0-9]+)",
        r"Label\s*:\s*([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        value = int(match.group(1))
        if value in allowed:
            return value

    matches = re.findall(r"\b([0-9]+)\b", text)
    for raw in reversed(matches):
        value = int(raw)
        if value in allowed:
            return value
    return None


def format_latent_cot_output(
    latent_logits: Sequence[float],
    label_space: SST5LabelSpace | Dict,
) -> str:
    if isinstance(label_space, dict):
        label_values = [int(v) for v in label_space["label_values"]]
        label_text_by_value = {
            int(k): str(v) for k, v in label_space.get("label_text_by_value", {}).items()
        }
    else:
        label_values = [int(v) for v in label_space.label_values]
        label_text_by_value = label_space.label_text_by_value

    scores = [(int(label_values[idx]), float(score)) for idx, score in enumerate(latent_logits)]
    scores_sorted = sorted(scores, key=lambda item: item[1], reverse=True)
    pred_label, pred_score = scores_sorted[0]
    runner_up_label, runner_up_score = scores_sorted[1] if len(scores_sorted) > 1 else (pred_label, pred_score)
    pred_text = label_text_by_value.get(pred_label, str(pred_label))
    runner_up_text = label_text_by_value.get(runner_up_label, str(runner_up_label))
    margin = pred_score - runner_up_score
    return (
        "Reasoning: The compressed SST-5 latent logits place the highest score on "
        f"{pred_label} ({pred_text}), ahead of {runner_up_label} ({runner_up_text}) "
        f"by a margin of {margin:.4f}.\n"
        f"Final label: {pred_label}"
    )


def build_content_token_space(
    tokenizer,
    train_jsonl: str | Path,
    label_space: SST5LabelSpace,
    label_candidate_spec: CandidateTokenSpec,
    min_token_count: int = 1,
    max_k: Optional[int] = None,
    include_label_text: bool = True,
    teacher_prompt_token_ids_by_record: Optional[Sequence[Sequence[int]]] = None,
    teacher_cot_token_ids_by_record: Optional[Sequence[Sequence[int]]] = None,
    teacher_next_token_ids_by_record: Optional[Sequence[Sequence[int]]] = None,
) -> SST5ContentTokenSpace:
    if min_token_count <= 0:
        raise ValueError(f"min_token_count must be positive, got {min_token_count}")

    records = read_jsonl(train_jsonl)
    validate_records_against_label_space(records, label_space, split="train")
    counts: Counter[int] = Counter()
    source_fields = ["text", "label"]
    if include_label_text:
        source_fields.append("label_text")
    if teacher_prompt_token_ids_by_record is not None:
        source_fields.append("teacher_prompt")
    if teacher_cot_token_ids_by_record is not None:
        source_fields.append("teacher_cot")
    if teacher_next_token_ids_by_record is not None:
        source_fields.append("teacher_next_token_target")

    forced_label_ids = {int(token_id) for token_id in label_candidate_spec.candidate_token_ids}
    forced_label_order = {
        int(token_id): idx
        for idx, token_id in enumerate(label_candidate_spec.candidate_token_ids)
    }
    forced_teacher_ids: set[int] = set()
    forced_teacher_order: Dict[int, int] = {}
    forced_source_ids: set[int] = set()
    forced_source_order: Dict[int, int] = {}

    def add_source_id(token_id: int) -> None:
        token_id = int(token_id)
        counts[token_id] += 1
        if token_id not in forced_source_ids:
            forced_source_order[token_id] = len(forced_source_order)
        forced_source_ids.add(token_id)

    def add_teacher_ids(rows: Optional[Sequence[Sequence[int]]], update_counts: bool = True) -> int:
        if rows is None:
            return 0
        total = 0
        for row in rows:
            for raw_token_id in row:
                token_id = int(raw_token_id)
                if update_counts:
                    counts[token_id] += 1
                total += 1
                if token_id not in forced_teacher_ids:
                    forced_teacher_order[token_id] = len(forced_teacher_order)
                forced_teacher_ids.add(token_id)
        return total

    for record in records:
        text_ids = tokenizer.encode(str(record["text"]), add_special_tokens=False)
        for token_id in text_ids:
            add_source_id(int(token_id))
        label = int(record["label"])
        label_pos = label_space.label_values.index(label)
        add_source_id(int(label_candidate_spec.candidate_token_ids[label_pos]))
        if include_label_text:
            label_text_ids = tokenizer.encode(str(record.get("label_text", "")), add_special_tokens=False)
            for token_id in label_text_ids:
                add_source_id(int(token_id))

    teacher_prompt_token_count = add_teacher_ids(teacher_prompt_token_ids_by_record)
    teacher_cot_token_count = add_teacher_ids(teacher_cot_token_ids_by_record)
    same_teacher_rows = teacher_next_token_ids_by_record is teacher_cot_token_ids_by_record
    teacher_next_token_count = add_teacher_ids(
        teacher_next_token_ids_by_record,
        update_counts=not same_teacher_rows,
    )

    # Force label tokens and teacher next-token targets into the closed K-space.
    # CE is token-level over all K positions, not a 5-way sentiment CE.
    for token_id in label_candidate_spec.candidate_token_ids:
        counts[int(token_id)] += 1

    forced_ids = forced_label_ids | forced_teacher_ids | forced_source_ids
    selected = [
        (int(token_id), int(count))
        for token_id, count in counts.items()
        if int(count) >= min_token_count or int(token_id) in forced_ids
    ]
    selected.sort(
        key=lambda item: (
            0
            if item[0] in forced_label_ids
            else (1 if item[0] in forced_teacher_ids else 2),
            forced_label_order.get(
                item[0],
                forced_teacher_order.get(item[0], forced_source_order.get(item[0], 0)),
            ),
            -item[1],
            item[0],
        )
    )

    if max_k is not None:
        if max_k < len(forced_ids):
            raise ValueError(
                f"max_k={max_k} is smaller than the number of forced K tokens "
                f"{len(forced_ids)} from labels and teacher targets"
            )
        forced = [item for item in selected if item[0] in forced_ids]
        non_forced = [item for item in selected if item[0] not in forced_ids]
        selected = forced + non_forced[: max(0, int(max_k) - len(forced))]

    candidate_token_ids = [token_id for token_id, _ in selected]
    token_counts = [count for _, count in selected]
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("Duplicate candidate token ids while building SST-5 content K-space")

    token_to_index = {int(token_id): idx for idx, token_id in enumerate(candidate_token_ids)}
    label_candidate_indices = []
    for token_id in label_candidate_spec.candidate_token_ids:
        token_id = int(token_id)
        if token_id not in token_to_index:
            raise ValueError(f"Label token id {token_id} was not included in content token space")
        label_candidate_indices.append(int(token_to_index[token_id]))

    label_to_candidate_index = {
        int(label): int(label_candidate_indices[idx])
        for idx, label in enumerate(label_space.label_values)
    }
    label_index_set = set(label_candidate_indices)
    text_candidate_indices = [
        int(idx) for idx in range(len(candidate_token_ids)) if idx not in label_index_set
    ]
    if len(label_candidate_indices) + len(text_candidate_indices) != len(candidate_token_ids):
        raise ValueError("Label/text K-space partition does not cover all candidate indices")

    candidate_texts = [
        tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in candidate_token_ids
    ]

    return SST5ContentTokenSpace(
        candidate_token_ids=candidate_token_ids,
        candidate_texts=candidate_texts,
        token_counts=token_counts,
        label_values=[int(v) for v in label_space.label_values],
        label_token_ids=[int(v) for v in label_candidate_spec.candidate_token_ids],
        label_candidate_indices=label_candidate_indices,
        text_candidate_indices=text_candidate_indices,
        label_to_candidate_index=label_to_candidate_index,
        label_text_by_value=label_space.label_text_by_value,
        num_train_records=label_space.num_train_records,
        source_fields=source_fields,
        teacher_prompt_token_count=int(teacher_prompt_token_count),
        teacher_cot_token_count=int(teacher_cot_token_count),
        teacher_next_token_count=int(teacher_next_token_count),
        frozen=True,
        min_token_count=int(min_token_count),
        max_k=None if max_k is None else int(max_k),
        train_text_label_sha256=label_space.train_text_label_sha256,
    )


def _encode_single_candidate(tokenizer, candidate_text: str) -> List[int]:
    return list(tokenizer.encode(candidate_text, add_special_tokens=False))


def choose_candidate_tokens(tokenizer, label_values: Sequence[int]) -> CandidateTokenSpec:
    labels = [int(v) for v in label_values]
    variants: List[Tuple[str, List[str], bool]] = [
        (
            "leading_space_digits",
            [f"{DEFAULT_LABEL_CANDIDATE_PREFIX}{v}" for v in labels],
            False,
        ),
        ("bare_digits_after_prompt_space", [str(v) for v in labels], True),
    ]

    diagnostics: List[Dict] = []
    for name, candidate_texts, prompt_trailing_space in variants:
        encoded = [_encode_single_candidate(tokenizer, text) for text in candidate_texts]
        diagnostics.append({"variant": name, "candidate_texts": candidate_texts, "ids": encoded})
        if all(len(ids) == 1 for ids in encoded):
            token_ids = [int(ids[0]) for ids in encoded]
            if len(set(token_ids)) != len(token_ids):
                raise ValueError(f"Candidate token ids are not unique for {name}: {token_ids}")
            return CandidateTokenSpec(
                candidate_texts=candidate_texts,
                candidate_token_ids=token_ids,
                prompt_trailing_space=prompt_trailing_space,
                variant_name=name,
            )

    raise ValueError(
        "Could not find a single-token numeric candidate variant. "
        f"Tokenizer diagnostics: {diagnostics}"
    )


def safe_name(value: str) -> str:
    value = value.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "teacher"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose one of {sorted(table)}")
    return table[dtype_name]


def load_pt(path: str | Path) -> Dict:
    path = project_path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def logits_cache_summary(obj: Dict) -> Dict:
    if "input_logitc_K" in obj:
        rows = obj["input_logitc_K"]
        if torch.is_tensor(rows):
            num_samples = int(rows.shape[0])
            k = int(rows.shape[-1])
            lengths = [int(rows.shape[1])] * num_samples
            shape = list(rows.shape)
        else:
            num_samples = len(rows)
            k = int(rows[0].shape[-1]) if rows else int(obj.get("k", 0))
            lengths = [int(row.shape[0]) for row in rows]
            shape = [[int(v) for v in row.shape] for row in rows[:3]]
        return {
            "cache_format": obj.get("cache_format", "token_sequence_k_space_v1"),
            "num_samples": num_samples,
            "num_tokens": int(sum(lengths)),
            "min_sequence_tokens": int(min(lengths)) if lengths else 0,
            "max_sequence_tokens": int(max(lengths)) if lengths else 0,
            "k": k,
            "input_logitc_K_shape": shape,
            "target_k_indices_present": "target_k_indices" in obj,
            "candidate_token_ids": [int(v) for v in obj.get("candidate_token_ids", [])],
            "candidate_texts": [str(v) for v in obj.get("candidate_texts", [])],
            "k_frozen": bool(obj.get("k_frozen", obj.get("content_token_space", {}).get("frozen", False))),
        }
    teacher_logits = obj["teacher_logits"]
    labels = obj["labels"]
    return {
        "num_samples": int(teacher_logits.shape[0]),
        "k": int(teacher_logits.shape[1]),
        "teacher_logits_shape": list(teacher_logits.shape),
        "labels_shape": list(labels.shape),
        "label_values": [int(v) for v in obj.get("label_values", [])],
        "candidate_token_ids": [int(v) for v in obj.get("candidate_token_ids", [])],
        "candidate_texts": [str(v) for v in obj.get("candidate_texts", [])],
        "label_candidate_indices": [int(v) for v in obj.get("label_candidate_indices", [])],
        "text_candidate_indices": [int(v) for v in obj.get("text_candidate_indices", [])],
        "k_label": int(obj.get("k_label", len(obj.get("label_candidate_indices", [])))),
        "k_text": int(obj.get("k_text", len(obj.get("text_candidate_indices", [])))),
    }


def ensure_k_tensor(name: str, tensor: torch.Tensor, k: int) -> None:
    if tensor.ndim == 1:
        ok = tensor.shape[0] == k
    else:
        ok = tensor.ndim in (2, 3) and tensor.shape[-1] == k
    if not ok:
        raise ValueError(
            f"{name} must have shape [K], [B,K], or [B,T,K] with K={k}; got {list(tensor.shape)}"
        )
