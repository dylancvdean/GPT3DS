# GPT3DS

## Good Petite Transformer with 3 Degrees of Self-attention

Does not stand for anything else.

GPT3DS is a tiny looped-transformer language model and Nintendo New 3DS
runtime. The repo contains the PyTorch training/export scripts, tokenizer
assets, quantized model weights, and a libctru C application that runs
generation directly on the handheld.

This is mostly an experiment in fitting a real chat-ish language model into the
New 3DS memory and CPU budget. It's a toy model and seems to only know 3 jokes, but it works.

## Download

[![QR code for GPT3DS v1.0.0 CIA download](assets/gpt3ds-v1.0.0-cia-qr.svg)](https://github.com/dylancvdean/GPT3DS/releases/download/v1.0.0/GPT3DS.cia)

[Download GPT3DS.cia v1.0.0](https://github.com/dylancvdean/GPT3DS/releases/download/v1.0.0/GPT3DS.cia)

## Project layout

- `train.py` - PyTorch pretraining/SFT script for the looped transformer
- `pretrain.sh`, `sft.sh` - example training entry points
- `3ds/export_3ds.py` - exports a checkpoint to 3DS runtime weights
- `3ds/source/` - C runtime, model forward pass, tokenizer, and app UI
- `3ds/romfs/` - ROMFS-baked weights, tokenizer, and generated model header
- `tokenizer_qwends_sft.json`, `tokenizer_qwends.cbin` - byte-level BPE tokenizer

## Model

- About 11.6M parameters
- 3 unique transformer blocks reused for 6 loops, for 18 effective layers
- `d_model=512`, `n_heads=8`, `head_dim=64`
- MLP hidden size 2048 with ReLU
- RMSNorm, learned position embeddings, and a causal depthwise conv after
  embeddings (which I have found to improve the performance of small looped transformer LLMs)
- Tied input/output embeddings
- 128 token context length
- 4096 token byte-level BPE vocabulary + 3 special tokens (`<u>`, `<a>`, and `<long>`)
- Pretrained on a subset of FineWeb-Edu, SFT on SmolTalk + synthetic data created for this project

## Runtime

The 3DS runtime is written in C against libctru. Model weights and tokenizer
data are baked into ROMFS so the app can be distributed as a single installable
file.

- Linear weights are symmetric per-row int8
- Embeddings are stored as fp16 and converted to fp32 in software
- Biases, norm weights, KV cache, activations, and logits use fp32
- Expected memory use is around 25 MB on device
- Inference is currently single-threaded

## Building the 3DS app

Install devkitPro with devkitARM and libctru, then build from the `3ds`
directory:

```bash
cd 3ds
export DEVKITPRO=/opt/devkitpro
export DEVKITARM="$DEVKITPRO/devkitARM"
export PATH="$DEVKITARM/bin:$DEVKITPRO/tools/bin:$PATH"
make
```

This produces `3ds/GPT3DS.3dsx` and, when `makerom` is available,
`3ds/GPT3DS.cia`.

## App controls

- `A` opens the software keyboard and starts generation after input
- `START` exits the app
- `X` resets context
- D-Pad can be used to adjust sampling
- Top screen shows chat history
- Bottom screen shows status
- 3D UI supported
