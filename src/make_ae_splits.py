import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("sample_id", "logits_path", "answer_len"):
                if key not in record:
                    raise KeyError(f"Missing key '{key}' on line {line_idx}: {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_write_main_alias(path: Path, records: list[dict]) -> Path | None:
    aliases = {
        "ae_train.jsonl": "main_ae_train.jsonl",
        "ae_val.jsonl": "main_ae_val.jsonl",
    }
    alias_name = aliases.get(path.name)
    if alias_name is None:
        return None

    alias_path = path.with_name(alias_name)
    if alias_path.resolve() == path.resolve():
        return None
    write_jsonl(alias_path, records)
    return alias_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create question-level AE train/val splits.")
    parser.add_argument("--input", type=str, default="data/meta/main_train.jsonl")
    parser.add_argument("--train-out", type=str, default="data/meta/ae_train.jsonl")
    parser.add_argument("--val-out", type=str, default="data/meta/ae_val.jsonl")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    train_out = resolve_path(args.train_out)
    val_out = resolve_path(args.val_out)

    records = read_jsonl(input_path)
    shuffled = list(records)
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)

    if args.val_size < 0:
        raise ValueError("--val-size must be >= 0")
    if args.val_size > len(shuffled):
        raise ValueError(f"--val-size={args.val_size} exceeds total records={len(shuffled)}")

    train_size = args.train_size
    if train_size is None:
        train_size = len(shuffled) - args.val_size
    if train_size < 0:
        raise ValueError("--train-size must be >= 0")
    if train_size + args.val_size > len(shuffled):
        raise ValueError(
            f"train_size + val_size = {train_size + args.val_size} exceeds total records={len(shuffled)}"
        )

    val_records = shuffled[: args.val_size]
    train_records = shuffled[args.val_size : args.val_size + train_size]

    write_jsonl(train_out, train_records)
    write_jsonl(val_out, val_records)
    train_alias = maybe_write_main_alias(train_out, train_records)
    val_alias = maybe_write_main_alias(val_out, val_records)

    print(f"total records: {len(records)}")
    print(f"train records: {len(train_records)}")
    print(f"val records: {len(val_records)}")
    print(f"seed: {args.seed}")
    print(f"train output: {train_out}")
    print(f"val output: {val_out}")
    if train_alias is not None:
        print(f"train alias: {train_alias}")
    if val_alias is not None:
        print(f"val alias: {val_alias}")
    print("final test remains data/meta/main_test.jsonl")


if __name__ == "__main__":
    main()
