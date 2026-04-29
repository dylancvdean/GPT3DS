#!/usr/bin/env python3
"""
export_3ds.py

Export the PyTorch checkpoint to a quantized binary + C header for the 3DS runtime.

Weights are quantized to symmetric per-row int8 to save memory and use integer ALU.
Embedding tables are kept as fp16 (converted to fp32 at load time).
Biases and norm weights stay fp32.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def write_u32(f, v: int) -> None:
    f.write(struct.pack("<I", v))


def write_f32(f, v: float) -> None:
    f.write(struct.pack("<f", v))


@dataclass
class TensorInfo:
    name: str
    offset: int
    size_bytes: int
    shape: tuple[int, ...]
    dtype: str  # "fp32", "fp16", "int8"


def quantize_symmetric_int8(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a weight matrix to symmetric int8 per row.
    weight shape: [N, K] (out_features, in_features)
    Returns: w_int8 [N, K], scales [N] (fp32)
    """
    w = weight.detach().cpu().float().numpy()
    N, K = w.shape
    w_int8 = np.empty((N, K), dtype=np.int8)
    scales = np.empty(N, dtype=np.float32)
    for i in range(N):
        row = w[i]
        amax = np.max(np.abs(row))
        if amax < 1e-8:
            scales[i] = 1.0
            w_int8[i] = 0
        else:
            scales[i] = amax / 127.0
            w_int8[i] = np.clip(np.round(row / scales[i]), -127, 127).astype(np.int8)
    return w_int8, scales


def export_model(checkpoint_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"]
    model_config = ckpt.get("model_config", {})

    vocab_size = model_config.get("vocab_size", 4096)
    ctx_len = model_config.get("ctx_len", 128)
    d_model = model_config.get("d_model", 512)
    n_heads = model_config.get("n_heads", 8)
    unique_blocks = model_config.get("unique_blocks", 3)
    loops_per_pass = model_config.get("loops_per_pass", 6)
    mlp_mult = model_config.get("mlp_mult", 4)
    head_dim = d_model // n_heads
    mlp_hidden = d_model * mlp_mult
    bos_id = model_config.get("bos_id", 0)
    eos_id = model_config.get("eos_id", 1)
    pad_id = model_config.get("pad_id", 2)
    unk_id = model_config.get("unk_id", 3)

    bin_path = os.path.join(out_dir, "model_weights.bin")
    header_path = os.path.join(out_dir, "model_desc.h")

    tensors: list[TensorInfo] = []

    def append_tensor(f, name: str, arr: np.ndarray, dtype: str) -> None:
        offset = f.tell()
        if dtype == "fp32":
            f.write(arr.astype(np.float32).tobytes())
            size = arr.size * 4
        elif dtype == "fp16":
            f.write(arr.astype(np.float16).tobytes())
            size = arr.size * 2
        elif dtype == "int8":
            f.write(arr.astype(np.int8).tobytes())
            size = arr.size
        else:
            raise ValueError(f"Unknown dtype {dtype}")
        tensors.append(TensorInfo(name, offset, size, tuple(arr.shape), dtype))

    def append_q8_linear(f, name: str, weight: torch.Tensor) -> None:
        w_q8, scales = quantize_symmetric_int8(weight)
        append_tensor(f, f"{name}.weight_q8", w_q8, "int8")
        append_tensor(f, f"{name}.scales", scales, "fp32")

    with open(bin_path, "wb") as f:
        # Embeddings: fp16 to save space, loaded as fp32
        append_tensor(f, "token_emb", state["token_emb.weight"].cpu().float().numpy(), "fp16")
        append_tensor(f, "pos_emb", state["pos_emb.weight"].cpu().float().numpy(), "fp16")

        # Embedding conv: tiny, keep fp32
        conv_w = state["embed_conv.weight"].cpu().float().numpy()  # [512, 3]
        conv_b = state["embed_conv.bias"].cpu().float().numpy()    # [512]
        append_tensor(f, "embed_conv.weight", conv_w, "fp32")
        append_tensor(f, "embed_conv.bias", conv_b, "fp32")

        # Blocks
        for b in range(unique_blocks):
            prefix = f"blocks.{b}"
            # RMSNorm weights
            append_tensor(f, f"{prefix}.attn_norm.weight", state[f"{prefix}.attn_norm.weight"].cpu().float().numpy(), "fp32")
            append_tensor(f, f"{prefix}.mlp_norm.weight", state[f"{prefix}.mlp_norm.weight"].cpu().float().numpy(), "fp32")

            # Attention projections
            append_q8_linear(f, f"{prefix}.attn.q_proj", state[f"{prefix}.attn.q_proj.weight"])
            append_q8_linear(f, f"{prefix}.attn.k_proj", state[f"{prefix}.attn.k_proj.weight"])
            append_q8_linear(f, f"{prefix}.attn.v_proj", state[f"{prefix}.attn.v_proj.weight"])
            append_q8_linear(f, f"{prefix}.attn.o_proj", state[f"{prefix}.attn.o_proj.weight"])

            # MLP
            append_q8_linear(f, f"{prefix}.mlp.up", state[f"{prefix}.mlp.up.weight"])
            append_q8_linear(f, f"{prefix}.mlp.down", state[f"{prefix}.mlp.down.weight"])

        # Final norm + head bias
        append_tensor(f, "final_norm.weight", state["final_norm.weight"].cpu().float().numpy(), "fp32")
        append_tensor(f, "lm_head_bias", state["lm_head_bias"].cpu().float().numpy(), "fp32")

    # Generate C header
    total_size = os.path.getsize(bin_path)
    total_params = sum(math.prod(t.shape) for t in tensors)

    with open(header_path, "w") as f:
        f.write("// Auto-generated by export_3ds.py\n")
        f.write("#ifndef MODEL_DESC_H\n")
        f.write("#define MODEL_DESC_H\n\n")

        f.write(f"#define MODEL_VOCAB_SIZE {vocab_size}\n")
        f.write(f"#define MODEL_CTX_LEN {ctx_len}\n")
        f.write(f"#define MODEL_D_MODEL {d_model}\n")
        f.write(f"#define MODEL_N_HEADS {n_heads}\n")
        f.write(f"#define MODEL_HEAD_DIM {head_dim}\n")
        f.write(f"#define MODEL_UNIQUE_BLOCKS {unique_blocks}\n")
        f.write(f"#define MODEL_LOOPS_PER_PASS {loops_per_pass}\n")
        f.write(f"#define MODEL_MLP_MULT {mlp_mult}\n")
        f.write(f"#define MODEL_MLP_HIDDEN {mlp_hidden}\n")
        f.write(f"#define MODEL_BOS_ID {bos_id}\n")
        f.write(f"#define MODEL_EOS_ID {eos_id}\n")
        f.write(f"#define MODEL_PAD_ID {pad_id}\n")
        f.write(f"#define MODEL_UNK_ID {unk_id}\n")
        f.write(f"#define MODEL_N_LAYERS (MODEL_UNIQUE_BLOCKS * MODEL_LOOPS_PER_PASS)\n\n")

        f.write(f"#define MODEL_WEIGHTS_SIZE {total_size}\n")
        f.write(f"#define MODEL_TOTAL_PARAMS {total_params}\n\n")

        for t in tensors:
            name = t.name.replace(".", "_")
            f.write(f"/* {t.name} shape={t.shape} dtype={t.dtype} */\n")
            f.write(f"#define OFF_{name.upper()} {t.offset}\n")
            f.write(f"#define SIZE_{name.upper()} {t.size_bytes}\n")
            # Flattened size for element counts
            elems = math.prod(t.shape)
            f.write(f"#define ELEMS_{name.upper()} {elems}\n")
            for i, dim in enumerate(t.shape):
                f.write(f"#define DIM_{name.upper()}_{i} {dim}\n")
            f.write("\n")

        f.write("#endif // MODEL_DESC_H\n")

    print(f"Exported {len(tensors)} tensors.")
    print(f"Binary: {bin_path} ({total_size / 1024 / 1024:.2f} MB)")
    print(f"Header: {header_path}")
    print(f"Total parameters (including scales): {total_params:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt_fineweb.best.pt")
    parser.add_argument("--out_dir", default="3ds/romfs")
    args = parser.parse_args()
    export_model(args.checkpoint, args.out_dir)


if __name__ == "__main__":
    main()
