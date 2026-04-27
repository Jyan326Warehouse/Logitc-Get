import argparse
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


def load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_answer_logits(obj: Dict, path: Path) -> torch.Tensor:
    if "answer_logits" not in obj:
        raise KeyError(f"{path} is missing answer_logits")
    logits = obj["answer_logits"]
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if logits.ndim != 2:
        raise ValueError(f"{path} answer_logits must be [T, V], got {tuple(logits.shape)}")
    return logits


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


def load_token_list(token_list_json: str) -> Dict:
    path = Path(token_list_json)
    with open(path, "r", encoding="utf-8") as f:
        token_list = json.load(f)

    token_ids = [int(t) for t in token_list["token_ids"]]
    token_to_index = {str(k): int(v) for k, v in token_list["token_to_index"].items()}
    if len(token_ids) != len(token_to_index):
        raise ValueError(
            f"token_ids length {len(token_ids)} != token_to_index length {len(token_to_index)}"
        )
    return token_list


class GSMTokenListDataset(Dataset):
    """
    Project teacher full-vocab answer logits into the GSM token-list space.

    Each item is one answer-token position:
        input_logits_k: Tensor[K]
        target_idx: LongTensor scalar, index into token_ids
        token_id: original vocabulary token id
        source_file: source .pt path
        position: answer-token position inside the file
    """

    def __init__(
        self,
        logits_dir: str,
        token_list_json: str,
        max_samples: Optional[int] = None,
        skip_oov: bool = True,
        preload: bool = True,
        logits_dtype: torch.dtype = torch.float32,
        cache_size: int = 4,
    ):
        super().__init__()
        self.logits_dir = Path(logits_dir)
        self.token_list_json = Path(token_list_json)
        self.max_samples = max_samples
        self.skip_oov = skip_oov
        self.preload = preload
        self.logits_dtype = logits_dtype
        self.cache_size = cache_size

        if not self.logits_dir.exists():
            raise FileNotFoundError(f"Logits dir not found: {self.logits_dir}")

        token_list = load_token_list(str(self.token_list_json))
        self.token_ids: List[int] = [int(t) for t in token_list["token_ids"]]
        self.token_texts: List[str] = token_list.get("token_texts", [""] * len(self.token_ids))
        if len(self.token_texts) < len(self.token_ids):
            self.token_texts = self.token_texts + [""] * (len(self.token_ids) - len(self.token_texts))
        self.token_to_index: Dict[str, int] = {
            str(k): int(v) for k, v in token_list["token_to_index"].items()
        }
        self.token_ids_tensor = torch.tensor(self.token_ids, dtype=torch.long)
        self.k = len(self.token_ids)

        self.files = sorted(self.logits_dir.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found under {self.logits_dir}")

        self.total_positions = 0
        self.skipped_oov_count = 0
        self.evaluated_positions = 0
        self.samples: List[Dict] = []
        self.input_logits_chunks: List[torch.Tensor] = []
        self.target_idx_chunks: List[torch.Tensor] = []
        self.token_id_chunks: List[torch.Tensor] = []
        self.position_chunks: List[torch.Tensor] = []
        self.source_file_chunks: List[str] = []
        self.cum_preloaded_counts: List[int] = []
        self.index: List[Dict] = []
        self._file_cache: OrderedDict[str, Dict] = OrderedDict()

        if preload:
            self._preload_samples()
        else:
            self._build_lazy_index()

    @property
    def oov_rate(self) -> float:
        if self.total_positions == 0:
            return 0.0
        return float(self.skipped_oov_count) / float(self.total_positions)

    def _target_index_for_token(self, token_id: int) -> Optional[int]:
        value = self.token_to_index.get(str(int(token_id)))
        if value is None:
            return None
        return int(value)

    def _should_stop(self) -> bool:
        return self.max_samples is not None and self.evaluated_positions >= self.max_samples

    def _preload_samples(self) -> None:
        for path in self.files:
            obj = load_pt(path)
            answer_logits = get_answer_logits(obj, path)
            answer_ids = get_answer_token_ids(obj, path)
            if answer_logits.shape[0] != answer_ids.numel():
                raise ValueError(
                    f"{path} length mismatch: answer_logits T={answer_logits.shape[0]}, "
                    f"answer_token_ids T={answer_ids.numel()}"
                )

            keep_positions: List[int] = []
            keep_target_idx: List[int] = []
            keep_token_ids: List[int] = []
            for position, token_id_tensor in enumerate(answer_ids):
                if self._should_stop():
                    break
                self.total_positions += 1
                token_id = int(token_id_tensor.item())
                target_idx = self._target_index_for_token(token_id)
                if target_idx is None:
                    self.skipped_oov_count += 1
                    if self.skip_oov:
                        continue
                    target_idx = -1

                keep_positions.append(int(position))
                keep_target_idx.append(int(target_idx))
                keep_token_ids.append(int(token_id))
                self.evaluated_positions += 1

            if keep_positions:
                pos_tensor = torch.tensor(keep_positions, dtype=torch.long)
                input_logits_k = answer_logits.index_select(0, pos_tensor).index_select(
                    1, self.token_ids_tensor
                )
                self.input_logits_chunks.append(input_logits_k.to(self.logits_dtype).contiguous())
                self.target_idx_chunks.append(torch.tensor(keep_target_idx, dtype=torch.long))
                self.token_id_chunks.append(torch.tensor(keep_token_ids, dtype=torch.long))
                self.position_chunks.append(pos_tensor)
                self.source_file_chunks.append(str(path))
                running = len(self) if self.cum_preloaded_counts else 0
                self.cum_preloaded_counts.append(running + len(keep_positions))

            del obj, answer_logits, answer_ids
            if self._should_stop():
                return

    def _build_lazy_index(self) -> None:
        for path in self.files:
            obj = load_pt(path)
            answer_logits = get_answer_logits(obj, path)
            answer_ids = get_answer_token_ids(obj, path)
            if answer_logits.shape[0] != answer_ids.numel():
                raise ValueError(
                    f"{path} length mismatch: answer_logits T={answer_logits.shape[0]}, "
                    f"answer_token_ids T={answer_ids.numel()}"
                )

            for position, token_id_tensor in enumerate(answer_ids):
                if self._should_stop():
                    return
                self.total_positions += 1
                token_id = int(token_id_tensor.item())
                target_idx = self._target_index_for_token(token_id)
                if target_idx is None:
                    self.skipped_oov_count += 1
                    if self.skip_oov:
                        continue
                    target_idx = -1

                self.index.append(
                    {
                        "source_file": str(path),
                        "position": int(position),
                        "target_idx": int(target_idx),
                        "token_id": int(token_id),
                    }
                )
                self.evaluated_positions += 1

            del obj, answer_logits, answer_ids

    def _load_cached_file(self, source_file: str) -> Dict:
        if source_file in self._file_cache:
            obj = self._file_cache.pop(source_file)
            self._file_cache[source_file] = obj
            return obj

        obj = load_pt(Path(source_file))
        self._file_cache[source_file] = obj
        while len(self._file_cache) > self.cache_size:
            self._file_cache.popitem(last=False)
        return obj

    def __len__(self) -> int:
        if self.preload:
            return self.cum_preloaded_counts[-1] if self.cum_preloaded_counts else 0
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        if self.preload:
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
            chunk_idx = bisect_right(self.cum_preloaded_counts, idx)
            prev = 0 if chunk_idx == 0 else self.cum_preloaded_counts[chunk_idx - 1]
            local_idx = idx - prev
            return {
                "input_logits_k": self.input_logits_chunks[chunk_idx][local_idx],
                "target_idx": self.target_idx_chunks[chunk_idx][local_idx],
                "token_id": int(self.token_id_chunks[chunk_idx][local_idx].item()),
                "source_file": self.source_file_chunks[chunk_idx],
                "position": int(self.position_chunks[chunk_idx][local_idx].item()),
            }

        record = self.index[idx]
        obj = self._load_cached_file(record["source_file"])
        answer_logits = get_answer_logits(obj, Path(record["source_file"]))
        input_logits_k = answer_logits[record["position"]].index_select(0, self.token_ids_tensor)
        return {
            "input_logits_k": input_logits_k.to(self.logits_dtype),
            "target_idx": torch.tensor(record["target_idx"], dtype=torch.long),
            "token_id": int(record["token_id"]),
            "source_file": record["source_file"],
            "position": int(record["position"]),
        }


def build_gsm_tokenlist_dataset(
    logits_dir: str,
    token_list_json: str,
    max_samples: Optional[int] = None,
    skip_oov: bool = True,
    preload: bool = True,
    logits_dtype: torch.dtype = torch.float32,
) -> GSMTokenListDataset:
    return GSMTokenListDataset(
        logits_dir=logits_dir,
        token_list_json=token_list_json,
        max_samples=max_samples,
        skip_oov=skip_oov,
        preload=preload,
        logits_dtype=logits_dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check GSM token-list dataset projection.")
    parser.add_argument("--logits-dir", required=True)
    parser.add_argument("--token-list-json", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GSMTokenListDataset(
        logits_dir=args.logits_dir,
        token_list_json=args.token_list_json,
        max_samples=args.max_samples,
        preload=not args.no_preload,
    )
    print(
        {
            "num_samples": len(dataset),
            "k": dataset.k,
            "total_positions": dataset.total_positions,
            "skipped_oov_count": dataset.skipped_oov_count,
            "oov_rate": dataset.oov_rate,
        }
    )
    if len(dataset) > 0:
        sample = dataset[0]
        print(
            {
                "input_logits_k_shape": tuple(sample["input_logits_k"].shape),
                "target_idx": int(sample["target_idx"]),
                "token_id": int(sample["token_id"]),
                "source_file": sample["source_file"],
                "position": int(sample["position"]),
            }
        )


if __name__ == "__main__":
    main()
