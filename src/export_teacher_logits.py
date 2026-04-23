import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """Solve the following math problem step by step.

Question:
{question}

Answer:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地 Qwen3-8B 路径",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/gsm8k",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        choices=["main", "socratic"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="项目下 data 目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制导出样本数；默认导出整个 split",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Qwen 通常建议打开",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="已存在则跳过",
    )
    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def find_prompt_token_len(tokenizer, prompt_text: str, full_text: str, add_special_tokens=True):
    """
    目标：在 full_text 的 token 序列里，找到 prompt 部分占了多少 token。
    优先用 prefix 对齐；
    如果 prefix 不严格成立，再用 offset_mapping（仅 fast tokenizer 支持）回退。
    """
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    full_kwargs = dict(
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    # fast tokenizer 才支持 offset_mapping
    if getattr(tokenizer, "is_fast", False):
        full_kwargs["return_offsets_mapping"] = True

    full_enc = tokenizer(full_text, **full_kwargs)

    prompt_ids = prompt_enc["input_ids"][0]
    full_ids = full_enc["input_ids"][0]

    # 情况1：prompt tokenization 是 full tokenization 的前缀
    if len(full_ids) >= len(prompt_ids) and torch.equal(full_ids[: len(prompt_ids)], prompt_ids):
        return len(prompt_ids), full_enc

    # 情况2：使用 offset_mapping 回退
    if "offset_mapping" in full_enc:
        boundary = len(prompt_text)
        offsets = full_enc["offset_mapping"][0].tolist()

        prompt_len = 0
        for i, (start, end) in enumerate(offsets):
            # 特殊 token 通常 offset = (0, 0)
            if start == 0 and end == 0:
                prompt_len += 1
                continue

            # end <= boundary 说明该 token 还完全属于 prompt 部分
            if end <= boundary:
                prompt_len += 1
            else:
                break

        return prompt_len, full_enc

    raise ValueError(
        "无法可靠定位 prompt 与 answer 的 token 边界。"
        "当前 tokenizer 不是 fast tokenizer，且 prefix 对齐失败。"
    )


def ensure_dirs(output_root: Path, split: str):
    meta_dir = output_root / "meta"
    logits_dir = output_root / "logits" / split
    meta_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir, logits_dir


def resolve_dataset_split(dataset_name: str, split: str) -> str:
    dataset_aliases = {
        "openai/gsm8k": {
            "validation": "test",
        }
    }
    return dataset_aliases.get(dataset_name, {}).get(split, split)


def main():
    args = parse_args()

    resolved_split = resolve_dataset_split(args.dataset_name, args.split)

    project_root = Path(__file__).resolve().parent.parent
    output_root = (project_root / args.output_root).resolve()
    meta_dir, logits_dir = ensure_dirs(output_root, resolved_split)

    meta_path = meta_dir / f"{args.dataset_config}_{resolved_split}.jsonl"

    load_dtype = resolve_dtype(args.load_dtype)
    save_dtype = resolve_dtype(args.save_dtype)

    print("=" * 80)
    print("Loading tokenizer and model...")
    print(f"model_path   = {args.model_path}")
    print(f"dataset      = {args.dataset_name}/{args.dataset_config}")
    print(f"split        = {resolved_split}")
    print(f"output_root  = {output_root}")
    print(f"meta_path    = {meta_path}")
    print(f"load_dtype   = {load_dtype}")
    print(f"save_dtype   = {save_dtype}")
    print("=" * 80)

    if args.split != resolved_split:
        print(
            f"[INFO] Requested split '{args.split}' is not available for "
            f"{args.dataset_name}; using '{resolved_split}' instead."
        )

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

    # 兼容 device_map="auto"
    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=resolved_split)

    end_idx = len(dataset) if args.max_samples is None else min(len(dataset), args.start_idx + args.max_samples)
    indices = list(range(args.start_idx, end_idx))

    print(f"Total dataset size = {len(dataset)}")
    print(f"Export range       = [{args.start_idx}, {end_idx})")
    print()

    num_ok = 0
    num_skip = 0
    num_err = 0

    # 这里用 overwrite 模式，避免多次重复写旧索引
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        for idx in tqdm(indices, desc="Exporting teacher logits"):
            ex = dataset[idx]
            question = ex["question"]
            answer_text = ex["answer"]

            sample_id = f"gsm8k_{args.dataset_config}_{resolved_split}_{idx:06d}"
            logits_file = logits_dir / f"{sample_id}.pt"

            if args.skip_existing and logits_file.exists():
                num_skip += 1
                continue

            try:
                prompt_text = build_prompt(question)
                full_text = prompt_text + answer_text

                prompt_len, full_enc = find_prompt_token_len(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    full_text=full_text,
                    add_special_tokens=True,
                )

                input_ids = full_enc["input_ids"]          # [1, L]
                attention_mask = full_enc["attention_mask"]  # [1, L]

                full_len = input_ids.shape[1]
                answer_len = full_len - prompt_len

                if answer_len <= 0:
                    raise ValueError(f"answer_len <= 0, sample_id={sample_id}")

                input_ids_gpu = input_ids.to(model_device)
                attention_mask_gpu = attention_mask.to(model_device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids_gpu,
                        attention_mask=attention_mask_gpu,
                        use_cache=False,
                    )
                    logits = outputs.logits[0]  # [L, V]

                # full_ids[prompt_len : full_len] 是 answer token 本身
                answer_ids = input_ids[0, prompt_len:full_len].cpu()  # [T]

                # 预测这些 answer token 的 logits 来自前一位置：
                # logits[prompt_len-1 : full_len-1]
                if prompt_len < 1:
                    raise ValueError(f"prompt_len < 1, sample_id={sample_id}")

                answer_logits = logits[prompt_len - 1 : full_len - 1].detach().to(save_dtype).cpu()

                if answer_logits.shape[0] != answer_ids.shape[0]:
                    raise ValueError(
                        f"length mismatch: answer_logits={answer_logits.shape[0]}, "
                        f"answer_ids={answer_ids.shape[0]}, sample_id={sample_id}"
                    )

                save_obj = {
                    "sample_id": sample_id,
                    "answer_logits": answer_logits,      # [T, V]
                    "answer_ids": answer_ids,            # [T]
                    "full_input_ids": input_ids[0].cpu(),# [L]
                    "prompt_len": prompt_len,
                    "full_len": full_len,
                    "answer_len": answer_len,
                }

                torch.save(save_obj, logits_file)

                meta_record = {
                    "sample_id": sample_id,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "split": resolved_split,
                    "index": idx,
                    "question": question,
                    "answer_text": answer_text,
                    "prompt_text": prompt_text,
                    "logits_path": str(logits_file.relative_to(output_root)),
                    "prompt_len": prompt_len,
                    "full_len": full_len,
                    "answer_len": answer_len,
                }
                f_meta.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

                num_ok += 1

                # 及时释放显存
                del outputs, logits, answer_logits, input_ids_gpu, attention_mask_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                err_record = {
                    "sample_id": sample_id,
                    "index": idx,
                    "error": str(e),
                }
                print(f"[ERROR] {json.dumps(err_record, ensure_ascii=False)}")
                num_err += 1

    print()
    print("=" * 80)
    print("Done.")
    print(f"OK    : {num_ok}")
    print(f"SKIP  : {num_skip}")
    print(f"ERROR : {num_err}")
    print(f"Meta file written to: {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
