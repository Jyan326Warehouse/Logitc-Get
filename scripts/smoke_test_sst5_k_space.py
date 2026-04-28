from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sst5_common import CandidateTokenSpec, build_content_token_space, infer_label_space
from sst5_dataset import (
    SST5TeacherLogitsDataset,
    make_target_onehot_K,
    sst5_k_space_collate_fn,
)
from sst5_content_ae_model import SST5ContentAE, SST5ContentAEConfig
from train_sst5_content_ae import compute_k_space_loss
from export_sst5_teacher_logits import map_token_ids_to_k_indices_and_mask


class DummyTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {}
        self.id_to_token = {}

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id) + 1
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False):
        parts = text.strip().split()
        return [self.add_token(part) for part in parts]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(self.id_to_token.get(int(idx), f"<unk:{int(idx)}>") for idx in ids)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_cache(path: Path, token_space, target_rows, bad_target: bool = False) -> None:
    token_to_k = {
        int(token_id): int(idx)
        for idx, token_id in enumerate(token_space.candidate_token_ids)
    }
    input_logitc_K = []
    target_k_indices = []
    teacher_next_token_ids = []
    input_ids = []
    loss_mask = []
    labels = []
    texts = []
    label_texts = []
    indices = []
    teacher_cot_outputs = []
    for idx, target_ids in enumerate(target_rows):
        t = len(target_ids)
        k = token_space.k
        logits = torch.randn(t, k, dtype=torch.float32)
        targets = torch.tensor([token_to_k[int(token_id)] for token_id in target_ids], dtype=torch.long)
        if bad_target and idx == 0:
            targets[0] = k
        input_logitc_K.append(logits)
        target_k_indices.append(targets)
        teacher_next_token_ids.append(torch.tensor(target_ids, dtype=torch.long))
        input_ids.append(torch.tensor(target_ids, dtype=torch.long))
        loss_mask.append(torch.ones(t, dtype=torch.bool))
        labels.append(idx % 5)
        texts.append(f"text {idx}")
        label_texts.append("label_text")
        indices.append(idx)
        teacher_cot_outputs.append("Reasoning: dummy")
    torch.save(
        {
            "cache_format": "token_sequence_k_space_v1",
            "source": "smoke",
            "split": "train",
            "input_logitc_K": input_logitc_K,
            "target_k_indices": target_k_indices,
            "teacher_next_token_ids": teacher_next_token_ids,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "texts": texts,
            "label_texts": label_texts,
            "indices": indices,
            "teacher_cot_outputs": teacher_cot_outputs,
            "label_values": [0, 1, 2, 3, 4],
            "candidate_token_ids": token_space.candidate_token_ids,
            "candidate_texts": token_space.candidate_texts,
            "content_token_space": token_space.to_dict(),
            "k": token_space.k,
            "k_frozen": True,
        },
        path,
    )


def main() -> None:
    work_dir = PROJECT_ROOT / "outputs" / "smoke_sst5_k_space"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    data_dir = work_dir / "data"
    train_path = data_dir / "train.jsonl"
    rows = [
        {"text": "awful boring", "label": 0, "label_text": "very negative"},
        {"text": "bad dull", "label": 1, "label_text": "negative"},
        {"text": "okay mixed", "label": 2, "label_text": "neutral"},
        {"text": "good warm", "label": 3, "label_text": "positive"},
        {"text": "great excellent", "label": 4, "label_text": "very positive"},
    ]
    write_jsonl(train_path, rows)

    tokenizer = DummyTokenizer()
    label_space = infer_label_space(train_path, expected_num_labels=5)
    label_ids = [tokenizer.add_token(str(value)) for value in label_space.label_values]
    teacher_rows = [
        [tokenizer.add_token("Reasoning:"), tokenizer.add_token("awful"), tokenizer.add_token("0")],
        [tokenizer.add_token("Reasoning:"), tokenizer.add_token("bad"), tokenizer.add_token("1")],
        [tokenizer.add_token("Reasoning:"), tokenizer.add_token("mixed"), tokenizer.add_token("2")],
    ]
    prompt_rows = [
        [tokenizer.add_token("Prompt:"), tokenizer.add_token("awful"), tokenizer.add_token("review")],
        [tokenizer.add_token("Prompt:"), tokenizer.add_token("bad"), tokenizer.add_token("review")],
        [tokenizer.add_token("Prompt:"), tokenizer.add_token("mixed"), tokenizer.add_token("review")],
    ]
    next_rows = [prompt[1:] + target for prompt, target in zip(prompt_rows, teacher_rows)]
    label_spec = CandidateTokenSpec(
        candidate_texts=[str(value) for value in label_space.label_values],
        candidate_token_ids=label_ids,
        prompt_trailing_space=False,
        variant_name="dummy_digits",
    )
    token_space = build_content_token_space(
        tokenizer=tokenizer,
        train_jsonl=train_path,
        label_space=label_space,
        label_candidate_spec=label_spec,
        teacher_prompt_token_ids_by_record=prompt_rows,
        teacher_cot_token_ids_by_record=teacher_rows,
        teacher_next_token_ids_by_record=next_rows,
    )
    assert token_space.frozen is True
    for row in prompt_rows + teacher_rows + next_rows:
        for token_id in row:
            assert int(token_id) in token_space.candidate_token_ids

    cache_path = work_dir / "train.pt"
    build_cache(cache_path, token_space, teacher_rows)
    dataset = SST5TeacherLogitsDataset(cache_path, expected_num_labels=5)
    batch = sst5_k_space_collate_fn([dataset[0], dataset[1]], include_target_onehot=True)
    bsz, steps, k = batch["input_logitc_K"].shape
    assert batch["input_ids"].shape == (bsz, steps)
    assert batch["attention_mask"].shape == (bsz, steps)
    assert batch["target_k_indices"].shape == (bsz, steps)
    assert batch["target_onehot_K"].shape == (bsz, steps, k)
    assert batch["loss_mask"].dtype == torch.bool

    onehot = make_target_onehot_K(batch["target_k_indices"], batch["loss_mask"], k)
    assert onehot.shape == (bsz, steps, k)
    model = SST5ContentAE(
        SST5ContentAEConfig(
            k=k,
            fusion_hidden_dim=8,
            encoder_hidden_dim=8,
            decoder_hidden_dim=8,
            dropout=0.0,
        )
    )
    outputs = model(batch["input_logitc_K"])
    assert outputs["latent_logits"].shape == (bsz, steps, k)
    assert outputs["recon_logits"].shape == (bsz, steps, k)
    assert outputs["latent_logits_K"].shape == (bsz, steps, k)
    assert outputs["recon_logitc_K"].shape == (bsz, steps, k)

    losses = compute_k_space_loss(
        latent_logits_K=outputs["latent_logits"],
        recon_logitc_K=outputs["recon_logits"],
        input_logitc_K=batch["input_logitc_K"],
        target_k_indices=batch["target_k_indices"],
        loss_mask=batch["loss_mask"],
        ce_weight=0.5,
        kl_weight=0.5,
        temperature=1.0,
        verify_target_onehot=True,
    )
    expected = 0.5 * losses["loss_ce"] + 0.5 * losses["loss_kl"]
    assert torch.allclose(losses["loss"], expected)

    bad_cache_path = work_dir / "bad.pt"
    build_cache(bad_cache_path, token_space, teacher_rows, bad_target=True)
    try:
        SST5TeacherLogitsDataset(bad_cache_path, expected_num_labels=5)
    except ValueError as exc:
        assert "outside frozen K" in str(exc)
    else:
        raise AssertionError("OOV target did not raise")

    unknown_token_id = tokenizer.add_token("validation_only_oov")
    sequence = {
        "index": 99,
        "text": "held out",
        "label": 0,
        "label_text": "very negative",
    }
    val_oov_path = work_dir / "validation_oov_records.jsonl"
    mapped, mask, num_oov = map_token_ids_to_k_indices_and_mask(
        token_ids=[teacher_rows[0][0], unknown_token_id, teacher_rows[0][-1]],
        sequence=sequence,
        tokenizer=tokenizer,
        token_id_to_k_index={
            int(token_id): int(idx)
            for idx, token_id in enumerate(token_space.candidate_token_ids)
        },
        split="validation",
        oov_path=val_oov_path,
    )
    assert num_oov == 1
    assert mapped.tolist()[1] == -100
    assert mask.tolist() == [True, False, True]
    assert val_oov_path.exists()
    try:
        map_token_ids_to_k_indices_and_mask(
            token_ids=[unknown_token_id],
            sequence=sequence,
            tokenizer=tokenizer,
            token_id_to_k_index={
                int(token_id): int(idx)
                for idx, token_id in enumerate(token_space.candidate_token_ids)
            },
            split="train",
            oov_path=work_dir / "train_oov_records.jsonl",
        )
    except ValueError as exc:
        assert "TRAIN split" in str(exc)
    else:
        raise AssertionError("Train OOV did not raise")

    print(
        json.dumps(
            {
                "status": "ok",
                "k": k,
                "batch_shape": list(batch["input_logitc_K"].shape),
                "loss": float(losses["loss"].item()),
                "loss_ce": float(losses["loss_ce"].item()),
                "loss_kl": float(losses["loss_kl"].item()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
