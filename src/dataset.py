import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class TokenLogitsDataset(Dataset):
    """
    把导出的 teacher logits 数据集转换成 token-level AE 训练样本。

    当前每个 .pt 文件里大致包含：
        - answer_logits: [T, V]
        - answer_ids:    [T]
        - sample_id
        - prompt_len
        - full_len
        - answer_len

    这里会把每个 token 位置展开成一条样本：
        x = answer_logits[t]   # [V]

    返回字段：
        {
            "x": Tensor[V],              # AE 输入（也是重建目标）
            "target_id": LongTensor[],   # 该位置真实 token id（可选辅助分析）
            "sample_id": str,
            "position": int,             # 该 token 在 answer 内的相对位置
            "answer_len": int,
        }
    """

    def __init__(
        self,
        data_root: str = "data",
        dataset_config: str = "main",
        split: str = "test",
        logits_dtype: torch.dtype = torch.float32,
        cache_size: int = 8,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        self.project_root = Path(__file__).resolve().parent.parent
        self.data_root = (self.project_root / data_root).resolve()
        self.dataset_config = dataset_config
        self.split = split
        self.logits_dtype = logits_dtype
        self.cache_size = cache_size

        self.meta_path = self.data_root / "meta" / f"{dataset_config}_{split}.jsonl"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        self.records = self._load_meta(self.meta_path, max_samples=max_samples)
        if len(self.records) == 0:
            raise ValueError(f"No records loaded from {self.meta_path}")

        self.cum_token_counts = self._build_cum_token_counts(self.records)
        self.total_tokens = self.cum_token_counts[-1]

        # 简单 LRU cache：缓存最近读取过的 .pt 文件
        self._file_cache: OrderedDict[str, Dict] = OrderedDict()

    def _load_meta(self, meta_path: Path, max_samples: Optional[int]) -> List[Dict]:
        records: List[Dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                required_keys = ["sample_id", "logits_path", "answer_len"]
                for k in required_keys:
                    if k not in record:
                        raise KeyError(f"Missing key '{k}' in meta line {line_idx}: {meta_path}")

                answer_len = int(record["answer_len"])
                if answer_len <= 0:
                    continue

                records.append(record)

                if max_samples is not None and len(records) >= max_samples:
                    break

        return records

    def _build_cum_token_counts(self, records: List[Dict]) -> List[int]:
        cum = []
        running = 0
        for r in records:
            running += int(r["answer_len"])
            cum.append(running)
        return cum

    def _get_record_and_local_pos(self, global_index: int):
        if global_index < 0 or global_index >= self.total_tokens:
            raise IndexError(f"Index {global_index} out of range [0, {self.total_tokens})")

        record_idx = bisect_right(self.cum_token_counts, global_index)
        prev_cum = 0 if record_idx == 0 else self.cum_token_counts[record_idx - 1]
        local_pos = global_index - prev_cum
        record = self.records[record_idx]
        return record, local_pos

    def _load_pt_file(self, rel_path: str) -> Dict:
        if rel_path in self._file_cache:
            obj = self._file_cache.pop(rel_path)
            self._file_cache[rel_path] = obj
            return obj

        abs_path = self.data_root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Logits file not found: {abs_path}")

        obj = torch.load(abs_path, map_location="cpu")

        self._file_cache[rel_path] = obj
        while len(self._file_cache) > self.cache_size:
            self._file_cache.popitem(last=False)

        return obj

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, index: int):
        record, local_pos = self._get_record_and_local_pos(index)
        obj = self._load_pt_file(record["logits_path"])

        answer_logits = obj["answer_logits"]   # [T, V]
        answer_ids = obj["answer_ids"]         # [T]

        if local_pos >= answer_logits.shape[0]:
            raise IndexError(
                f"local_pos={local_pos} out of range for sample_id={record['sample_id']} "
                f"with answer_logits.shape[0]={answer_logits.shape[0]}"
            )

        x = answer_logits[local_pos].to(self.logits_dtype)  # [V]
        target_id = answer_ids[local_pos].long()

        return {
            "x": x,
            "target_id": target_id,
            "sample_id": record["sample_id"],
            "position": local_pos,
            "answer_len": int(record["answer_len"]),
        }


def build_token_logits_dataset(
    data_root: str = "data",
    dataset_config: str = "main",
    split: str = "test",
    logits_dtype: torch.dtype = torch.float32,
    cache_size: int = 8,
    max_samples: Optional[int] = None,
) -> TokenLogitsDataset:
    return TokenLogitsDataset(
        data_root=data_root,
        dataset_config=dataset_config,
        split=split,
        logits_dtype=logits_dtype,
        cache_size=cache_size,
        max_samples=max_samples,
    )