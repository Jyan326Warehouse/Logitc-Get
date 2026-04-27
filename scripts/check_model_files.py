import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a local Hugging Face causal LM directory looks complete."
    )
    parser.add_argument(
        "--model-path",
        default="~/autodl-fs/model/Qwen3-8B",
        help="Local model directory to inspect.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Use trust_remote_code when optional transformers checks are enabled.",
    )
    parser.add_argument(
        "--skip-transformers-check",
        action="store_true",
        help="Only inspect files; do not try AutoConfig/AutoTokenizer local loading.",
    )
    return parser.parse_args()


def human_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        head = f.read(128)
    return head.startswith(LFS_POINTER_PREFIX)


def file_status(path: Path) -> Dict:
    exists = path.exists()
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": size,
        "size": human_size(size),
        "is_lfs_pointer": is_lfs_pointer(path) if is_file else False,
    }


def print_check(ok: bool, message: str) -> None:
    mark = "[OK]" if ok else "[FAIL]"
    print(f"{mark} {message}")


def unique_ordered(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def find_weight_index(model_path: Path) -> Optional[Path]:
    candidates = [
        model_path / "model.safetensors.index.json",
        model_path / "pytorch_model.bin.index.json",
        model_path / "tf_model.h5.index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_single_weight_files(model_path: Path) -> List[Path]:
    patterns = [
        "*.safetensors",
        "pytorch_model*.bin",
        "model*.bin",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(model_path.glob(pattern))
    return sorted(path for path in set(files) if path.is_file())


def check_basic_files(model_path: Path) -> Tuple[bool, List[Dict]]:
    checks = []
    required = [
        "config.json",
        "tokenizer_config.json",
    ]
    for name in required:
        status = file_status(model_path / name)
        checks.append(status)
        print_check(status["exists"] and status["is_file"], f"{name}: {status['size']}")

    tokenizer_candidates = [
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "qwen.tiktoken",
    ]
    found_tokenizer = [name for name in tokenizer_candidates if (model_path / name).is_file()]
    print_check(
        bool(found_tokenizer),
        "tokenizer payload: " + (", ".join(found_tokenizer) if found_tokenizer else "not found"),
    )
    for name in found_tokenizer:
        checks.append(file_status(model_path / name))

    optional = ["generation_config.json", "special_tokens_map.json"]
    for name in optional:
        if (model_path / name).exists():
            status = file_status(model_path / name)
            checks.append(status)
            print_check(status["is_file"], f"{name}: {status['size']}")

    ok = all(item["exists"] and item["is_file"] for item in checks[: len(required)]) and bool(found_tokenizer)
    return ok, checks


def check_weight_files(model_path: Path) -> Tuple[bool, Dict]:
    index_path = find_weight_index(model_path)
    if index_path is not None:
        index = read_json(index_path)
        weight_map = index.get("weight_map", {})
        shard_names = unique_ordered(str(name) for name in weight_map.values())
        expected_total = index.get("metadata", {}).get("total_size")

        print_check(True, f"weight index: {index_path.name}")
        print(f"     referenced shards: {len(shard_names)}")
        print(f"     index tensor total_size: {human_size(expected_total)}")

        missing = []
        pointers = []
        tiny = []
        total_size = 0
        shard_statuses = []
        for shard_name in shard_names:
            path = model_path / shard_name
            status = file_status(path)
            shard_statuses.append(status)
            if not status["exists"]:
                missing.append(shard_name)
                continue
            total_size += int(status["size_bytes"] or 0)
            if status["is_lfs_pointer"]:
                pointers.append(shard_name)
            if int(status["size_bytes"] or 0) < 1024 * 1024:
                tiny.append(shard_name)

        print_check(not missing, f"all referenced shards exist ({len(shard_names) - len(missing)}/{len(shard_names)})")
        print_check(not pointers, f"no shard is a Git LFS pointer ({len(pointers)} pointers)")
        print_check(not tiny, f"no shard is suspiciously tiny ({len(tiny)} tiny files)")
        print(f"     local shard file total: {human_size(total_size)}")
        if expected_total:
            ratio = float(total_size) / float(expected_total)
            print(f"     local/index size ratio: {ratio:.4f}")

        ok = not missing and not pointers and not tiny and total_size > 0
        return ok, {
            "mode": "indexed",
            "index_path": str(index_path),
            "expected_tensor_total_size_bytes": expected_total,
            "local_shard_total_size_bytes": total_size,
            "num_referenced_shards": len(shard_names),
            "missing_shards": missing,
            "lfs_pointer_shards": pointers,
            "tiny_shards": tiny,
            "shards": shard_statuses,
        }

    weight_files = find_single_weight_files(model_path)
    print_check(bool(weight_files), f"single/non-indexed weight files found: {len(weight_files)}")
    pointers = [path.name for path in weight_files if is_lfs_pointer(path)]
    tiny = [path.name for path in weight_files if path.stat().st_size < 1024 * 1024]
    total_size = sum(path.stat().st_size for path in weight_files)
    print_check(not pointers, f"no weight file is a Git LFS pointer ({len(pointers)} pointers)")
    print_check(not tiny, f"no weight file is suspiciously tiny ({len(tiny)} tiny files)")
    print(f"     local weight file total: {human_size(total_size)}")
    return bool(weight_files) and not pointers and not tiny, {
        "mode": "single_or_glob",
        "local_weight_total_size_bytes": total_size,
        "weight_files": [file_status(path) for path in weight_files],
        "lfs_pointer_files": pointers,
        "tiny_files": tiny,
    }


def check_transformers_local(model_path: Path, trust_remote_code: bool) -> bool:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:
        print_check(False, f"transformers import failed: {exc}")
        return False

    ok = True
    try:
        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model_type = getattr(config, "model_type", "unknown")
        vocab_size = getattr(config, "vocab_size", "unknown")
        print_check(True, f"AutoConfig local load ok | model_type={model_type} | vocab_size={vocab_size}")
    except Exception as exc:
        print_check(False, f"AutoConfig local load failed: {exc}")
        ok = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        print_check(True, f"AutoTokenizer local load ok | vocab_size={len(tokenizer)}")
    except Exception as exc:
        print_check(False, f"AutoTokenizer local load failed: {exc}")
        ok = False

    return ok


def main() -> None:
    args = parse_args()
    model_path = Path(os.path.expanduser(args.model_path)).resolve()

    print("=" * 80)
    print(f"Model path: {model_path}")
    print("=" * 80)

    if not model_path.exists():
        print_check(False, "model directory exists")
        raise SystemExit(2)
    print_check(model_path.is_dir(), "model directory exists")

    usage = shutil.disk_usage(model_path)
    print(f"Disk total: {human_size(usage.total)}")
    print(f"Disk used : {human_size(usage.used)}")
    print(f"Disk free : {human_size(usage.free)}")
    print()

    print("Basic files")
    print("-" * 80)
    basic_ok, basic = check_basic_files(model_path)
    print()

    print("Weight files")
    print("-" * 80)
    weights_ok, weights = check_weight_files(model_path)
    print()

    transformers_ok = True
    if not args.skip_transformers_check:
        print("Transformers local load checks")
        print("-" * 80)
        transformers_ok = check_transformers_local(
            model_path=model_path,
            trust_remote_code=args.trust_remote_code,
        )
        print()

    summary = {
        "model_path": str(model_path),
        "basic_ok": basic_ok,
        "weights_ok": weights_ok,
        "transformers_ok": transformers_ok,
        "overall_ok": basic_ok and weights_ok and transformers_ok,
        "basic_files": basic,
        "weights": weights,
    }
    print("Summary")
    print("-" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
