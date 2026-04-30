#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec python train.py \
  --data_source jsonl \
  --task sft \
  --train_jsonl data/smoltalk_pair128_train.jsonl \
  --val_jsonl data/smoltalk_pair128_val.jsonl \
  --hf_conversations_field messages \
  --tokenizer tokenizer_qwends.json \
  --sft_chat_template compact \
  --sft_tokenizer_out tokenizer_qwends_sft.json \
  --init_checkpoint ckpt_fineweb.best.pt \
  --out ckpt_smoltalk_sft.pt \
  --ctx_len 128 \
  --d_model 512 \
  --n_heads 8 \
  --unique_blocks 3 \
  --loops_per_pass 6 \
  --mlp_mult 4 \
  --batch_size 8 \
  --steps 5000 \
  --dtype fp16 \
  "$@"
