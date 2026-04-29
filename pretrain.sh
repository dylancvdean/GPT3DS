#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec python train.py \
  --data_source huggingface \
  --task pretrain \
  --hf_train_dataset HuggingFaceFW/fineweb-edu \
  --train_tokenizer \
  --tokenizer_out tokenizer_qwends.json \
  --export_tokenizer_cbin tokenizer_qwends.cbin \
  --out ckpt_fineweb.pt \
  --vocab_size 4096 \
  --ctx_len 128 \
  --d_model 512 \
  --n_heads 8 \
  --unique_blocks 3 \
  --loops_per_pass 6 \
  --mlp_mult 4 \
  --batch_size 32 \
  --steps 500000 \
  --lr 3e-4 \
  --warmup_steps 10000 \
  --min_lr_ratio 0.1 \
  --eval_every 5000 \
  --save_every 25000 \
  --sample_every 5000 \
  --dtype fp16 \
  "$@"
