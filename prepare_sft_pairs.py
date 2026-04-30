#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from train import (
    add_chat_special_tokens,
    encode_text_without_specials,
    extract_message_role_and_text,
    is_assistant_role,
    load_bpe_tokenizer,
    load_local_env_file,
    normalize_sft_role,
)


def is_user_role(raw_role: Any) -> bool:
    return normalize_sft_role(raw_role) == "User"


def write_pair_dataset(
    split: str,
    out_path: Path,
    tokenizer: Any,
    max_pair_input_tokens: int,
    max_assistant_tokens: int | None,
    cache_dir: str,
    seed: int,
    limit: int | None,
) -> tuple[int, int]:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dataset = load_dataset(
        "HuggingFaceTB/smoltalk",
        "all",
        split=split,
        cache_dir=cache_dir,
        token=token,
    )
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    considered = 0
    with out_path.open("w", encoding="utf-8") as file:
        for row_index, example in enumerate(dataset):
            if row_index > 0 and row_index % 100_000 == 0:
                print(
                    f"{split}: scanned {row_index:,} rows | "
                    f"kept {kept:,} / considered {considered:,}",
                    flush=True,
                )
            messages = example.get("messages")
            if not isinstance(messages, list):
                continue

            clean_messages: list[tuple[Any, str, int]] = []
            for message in messages:
                role, text = extract_message_role_and_text(message)
                if text is not None:
                    clean_messages.append((role, text, len(encode_text_without_specials(tokenizer, text))))

            for turn_index in range(1, len(clean_messages)):
                role, assistant_text, assistant_len = clean_messages[turn_index]
                prev_role, user_text, user_len = clean_messages[turn_index - 1]
                if not is_assistant_role(role) or not is_user_role(prev_role):
                    continue

                considered += 1
                if max_assistant_tokens is not None and assistant_len > max_assistant_tokens:
                    continue

                # Input side is <bos><u>{user}<a>{assistant}; <eos> is the final target.
                input_len = user_len + assistant_len + 3
                if input_len > max_pair_input_tokens:
                    continue

                output = {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "source": example.get("source"),
                    "row_index": row_index,
                    "turn_index": turn_index,
                    "input_tokens": input_len,
                    "assistant_tokens": assistant_len,
                }
                file.write(json.dumps(output, ensure_ascii=False) + "\n")
                kept += 1

    return kept, considered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer_qwends.json")
    parser.add_argument("--train_out", default="data/smoltalk_pairs_train.jsonl")
    parser.add_argument("--val_out", default="data/smoltalk_pairs_val.jsonl")
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--max_pair_input_tokens", type=int, default=128)
    parser.add_argument("--max_assistant_tokens", type=int, default=None)
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env_file(".env")

    tokenizer = load_bpe_tokenizer(args.tokenizer)
    add_chat_special_tokens(tokenizer)

    train_kept, train_considered = write_pair_dataset(
        "train",
        Path(args.train_out),
        tokenizer,
        max_pair_input_tokens=min(args.max_pair_input_tokens, args.ctx_len),
        max_assistant_tokens=args.max_assistant_tokens,
        cache_dir=args.cache_dir,
        seed=args.seed,
        limit=args.train_limit,
    )
    val_kept, val_considered = write_pair_dataset(
        "test",
        Path(args.val_out),
        tokenizer,
        max_pair_input_tokens=min(args.max_pair_input_tokens, args.ctx_len),
        max_assistant_tokens=args.max_assistant_tokens,
        cache_dir=args.cache_dir,
        seed=args.seed,
        limit=args.val_limit,
    )

    print(f"train kept {train_kept:,} / considered {train_considered:,}")
    print(f"val kept   {val_kept:,} / considered {val_considered:,}")
    print(f"wrote {args.train_out}")
    print(f"wrote {args.val_out}")


if __name__ == "__main__":
    main()
