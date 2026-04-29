#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec python train.py \
  --data_source huggingface \
  --task sft \
  --hf_train_dataset HuggingFaceTB/smoltalk \
  --hf_train_subset all \
  --hf_train_split train \
  --hf_val_dataset HuggingFaceTB/smoltalk \
  --hf_val_subset all \
  --hf_val_split test \
  --hf_conversations_field messages \
  --hf_no_streaming \
  --tokenizer tokenizer_qwends.json \
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
