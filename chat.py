#!/usr/bin/env python3
"""Interactive REPL for the SFT model using the compact chat template.

Usage:
    python chat.py
    python chat.py --checkpoint ckpt_smoltalk_sft.best.pt --temperature 0.7
    python chat.py --tokenizer tokenizer_qwends_sft.json

In-session commands: /reset, /quit (or Ctrl-D / Ctrl-C).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from train import (
    ASSISTANT_CHAT_TOKEN,
    LoopedTransformerLm,
    ModelConfig,
    USER_CHAT_TOKEN,
    encode_text_without_specials,
    get_autocast_dtype,
    get_device,
    load_bpe_tokenizer,
    special_id,
)


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[LoopedTransformerLm, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["model_config"]
    config = ModelConfig(**cfg_dict)
    model = LoopedTransformerLm(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt_smoltalk_sft.pt")
    parser.add_argument("--tokenizer", default="tokenizer_qwends_sft.json")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.tokenizer):
        sys.exit(f"tokenizer not found: {args.tokenizer}")

    device = get_device(args.device)
    print(f"loading {args.checkpoint} on {device}")
    model, cfg = build_model_from_ckpt(args.checkpoint, device)

    tokenizer = load_bpe_tokenizer(args.tokenizer)
    bos_id = cfg["bos_id"]
    eos_id = cfg["eos_id"]
    user_id = special_id(tokenizer, USER_CHAT_TOKEN)
    asst_id = special_id(tokenizer, ASSISTANT_CHAT_TOKEN)
    ctx_len = cfg["ctx_len"]

    autocast_dtype = get_autocast_dtype(args.dtype)
    use_autocast = device.type == "cuda" and autocast_dtype != torch.float32

    history: list[int] = [bos_id]

    print(f"vocab={cfg['vocab_size']} ctx={ctx_len} "
          f"params~{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("type /reset to clear history, /quit to exit\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            history = [bos_id]
            print("(history cleared)\n")
            continue

        turn_ids = [user_id, *encode_text_without_specials(tokenizer, user_text), asst_id]
        history.extend(turn_ids)

        # Sliding window: keep only the last ctx_len tokens (model truncates anyway).
        if len(history) > ctx_len:
            history = history[-ctx_len:]

        idx = torch.tensor([history], dtype=torch.long, device=device)
        prompt_len = idx.size(1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            generated = model.generate(
                idx,
                max_new_tokens=min(args.max_new_tokens, ctx_len - prompt_len),
                temperature=args.temperature,
                top_k=args.top_k,
                eos_id=eos_id,
            )

        new_ids = generated[0, prompt_len:].tolist()
        if new_ids and new_ids[-1] == eos_id:
            new_ids = new_ids[:-1]

        # Strip any stray special markers from display only.
        display_ids = [t for t in new_ids if t not in (bos_id, eos_id, user_id, asst_id)]
        reply = tokenizer.decode(display_ids, skip_special_tokens=True)
        print(f"bot> {reply}\n")

        history.extend(new_ids)
        if len(history) > ctx_len:
            history = history[-ctx_len:]


if __name__ == "__main__":
    main()
