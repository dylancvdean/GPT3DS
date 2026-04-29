#!/usr/bin/env python3
# train.py
#
# Tiny looped-transformer trainer for a 3DS-target language model.
#
# Architecture:
#   - Byte-level BPE tokenizer
#   - Learned absolute position embeddings
#   - Causal depthwise k=3 conv after token+position embeddings
#   - A small set of unique transformer blocks applied in loops
#   - RMSNorm
#   - Plain causal self-attention
#   - ReLU MLP
#   - Tied input/output embeddings
#
# Install:
#   pip install torch tokenizers datasets
#
# FineWeb-Edu streaming pretrain without downloading the full corpus:
#   python train.py \
#     --data_source huggingface \
#     --task pretrain \
#     --hf_train_dataset HuggingFaceFW/fineweb-edu \
#     --hf_train_subset sample-10BT \
#     --train_tokenizer \
#     --tokenizer_out tokenizer_qwends.json \
#     --export_tokenizer_cbin tokenizer_qwends.cbin \
#     --out ckpt_fineweb.pt \
#     --vocab_size 4096 \
#     --ctx_len 128 \
#     --d_model 512 \
#     --n_heads 8 \
#     --unique_blocks 3 \
#     --loops_per_pass 6 \
#     --batch_size 128 \
#     --steps 20000 \
#     --dtype fp16
#
# OpenHermes SFT from the pretrained checkpoint:
#   python train.py \
#     --data_source huggingface \
#     --task sft \
#     --hf_train_dataset teknium/OpenHermes-2.5 \
#     --hf_val_dataset teknium/OpenHermes-2.5 \
#     --tokenizer tokenizer_qwends.json \
#     --init_checkpoint ckpt_fineweb.best.pt \
#     --out ckpt_openhermes_sft.pt \
#     --ctx_len 128 \
#     --d_model 512 \
#     --n_heads 8 \
#     --unique_blocks 3 \
#     --loops_per_pass 6 \
#     --batch_size 64 \
#     --steps 5000 \
#     --dtype fp16

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


BOS_TEXT = "<bos>"
EOS_TEXT = "<eos>"
PAD_TEXT = "<pad>"
UNK_TEXT = "<unk>"

SYSTEM_LABEL = "System"
USER_LABEL = "User"
ASSISTANT_LABEL = "Assistant"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    ctx_len: int = 128
    d_model: int = 512
    n_heads: int = 8
    unique_blocks: int = 3
    loops_per_pass: int = 6
    mlp_mult: int = 4
    dropout: float = 0.0
    bos_id: int = 0
    eos_id: int = 1
    pad_id: int = 2
    unk_id: int = 3


@dataclass(frozen=True)
class TrainConfig:
    out: str
    tokenizer: str | None
    tokenizer_out: str
    train_tokenizer: bool
    export_tokenizer_cbin: str | None
    init_checkpoint: str | None
    data_source: str
    task: str
    train_text: str | None = None
    val_text: str | None = None
    hf_train_dataset: str | None = None
    hf_train_subset: str | None = None
    hf_train_split: str = "train"
    hf_val_dataset: str | None = None
    hf_val_subset: str | None = None
    hf_val_split: str | None = None
    hf_streaming: bool = True
    hf_text_field: str = "text"
    hf_conversations_field: str = "conversations"
    hf_system_prompt_field: str = "system_prompt"
    hf_shuffle_buffer: int = 10_000
    tokenizer_train_docs: int = 50_000
    train_docs_limit: int | None = None
    val_docs: int = 2_000
    train_buffer_tokens: int = 1_000_000
    refresh_buffer_tokens: int = 250_000
    sft_train_on_prompt: bool = False
    sft_min_example_tokens: int = 0
    sft_max_example_tokens: int | None = None
    hf_cache_dir: str = ".cache/huggingface"
    local_storage_budget_gb: float = 50.0
    vocab_size: int = 4096
    steps: int = 20_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1_000
    eval_every: int = 500
    eval_batches: int = 50
    save_every: int = 2_000
    sample_every: int = 500
    sample_prompt: str = "The answer is"
    sample_tokens: int = 120
    seed: int = 1337
    dtype: str = "fp16"
    device: str = "auto"


@dataclass
class EncodedSequence:
    token_ids: list[int]
    loss_mask: list[bool] | None = None


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TEXT))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    return tokenizer


def build_bpe_trainer(vocab_size: int) -> BpeTrainer:
    return BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[BOS_TEXT, EOS_TEXT, PAD_TEXT, UNK_TEXT],
        show_progress=True,
    )


def finalize_trained_tokenizer(tokenizer: Tokenizer, tokenizer_out: str) -> Tokenizer:
    bos_id = tokenizer.token_to_id(BOS_TEXT)
    eos_id = tokenizer.token_to_id(EOS_TEXT)
    if bos_id is None or eos_id is None:
        raise RuntimeError("failed to create BOS/EOS special tokens")

    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TEXT} $A {EOS_TEXT}",
        special_tokens=[(BOS_TEXT, bos_id), (EOS_TEXT, eos_id)],
    )

    ensure_parent_dir(tokenizer_out)
    tokenizer.save(tokenizer_out)
    return tokenizer


def train_bpe_tokenizer_from_file(train_text_path: str, tokenizer_out: str, vocab_size: int) -> Tokenizer:
    tokenizer = build_bpe_tokenizer()
    tokenizer.train([train_text_path], build_bpe_trainer(vocab_size))
    return finalize_trained_tokenizer(tokenizer, tokenizer_out)


def train_bpe_tokenizer_from_iterator(
    text_iter: Iterable[str],
    tokenizer_out: str,
    vocab_size: int,
    length: int | None = None,
) -> Tokenizer:
    tokenizer = build_bpe_tokenizer()
    tokenizer.train_from_iterator(text_iter, trainer=build_bpe_trainer(vocab_size), length=length)
    return finalize_trained_tokenizer(tokenizer, tokenizer_out)


def load_bpe_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def special_id(tokenizer: Tokenizer, token_text: str) -> int:
    token_id = tokenizer.token_to_id(token_text)
    if token_id is None:
        raise RuntimeError(f"missing special token {token_text}")
    return int(token_id)


def load_tokens(path: str, tokenizer: Tokenizer) -> torch.Tensor:
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    ids = tokenizer.encode(text).ids
    return torch.tensor(ids, dtype=torch.long)


def get_tokenizer_json(tokenizer_path: str) -> dict[str, Any]:
    with open(tokenizer_path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise RuntimeError("tokenizer JSON root was not an object")
    return value


def export_tokenizer_cbin(tokenizer_json_path: str, out_path: str) -> None:
    # Debug-friendly tokenizer export. For final 3DS runtime, I would later convert
    # merge strings into token IDs offline and make the C tokenizer even simpler.
    #
    # Format, little-endian:
    #   magic[8] = b"QDSBPE1\0"
    #   u32 vocab_size
    #   u32 merge_count
    #   u32 bos_id
    #   u32 eos_id
    #   u32 pad_id
    #   u32 unk_id
    #
    #   repeated vocab_size:
    #       u32 token_id
    #       u32 byte_len
    #       u8[byte_len] token_string_utf8
    #
    #   repeated merge_count:
    #       u32 rank
    #       u32 left_token_utf8_len
    #       u8[left_token_utf8_len]
    #       u32 right_token_utf8_len
    #       u8[right_token_utf8_len]

    data = get_tokenizer_json(tokenizer_json_path)
    model = data.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("tokenizer JSON has no model object")

    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise RuntimeError("tokenizer JSON did not look like BPE vocab/merges")

    vocab_by_id: list[tuple[int, str]] = []
    for token_text, token_id in vocab.items():
        if not isinstance(token_text, str) or not isinstance(token_id, int):
            raise RuntimeError("bad vocab entry")
        vocab_by_id.append((token_id, token_text))
    vocab_by_id.sort(key=lambda item: item[0])

    temp_tokenizer = load_bpe_tokenizer(tokenizer_json_path)
    bos_id = special_id(temp_tokenizer, BOS_TEXT)
    eos_id = special_id(temp_tokenizer, EOS_TEXT)
    pad_id = special_id(temp_tokenizer, PAD_TEXT)
    unk_id = special_id(temp_tokenizer, UNK_TEXT)

    ensure_parent_dir(out_path)
    with open(out_path, "wb") as file:
        file.write(b"QDSBPE1\0")
        file.write(
            struct.pack(
                "<IIIIII",
                len(vocab_by_id),
                len(merges),
                bos_id,
                eos_id,
                pad_id,
                unk_id,
            )
        )

        for token_id, token_text in vocab_by_id:
            encoded = token_text.encode("utf-8")
            file.write(struct.pack("<II", token_id, len(encoded)))
            file.write(encoded)

        for rank, merge_value in enumerate(merges):
            if isinstance(merge_value, str):
                pieces = merge_value.split()
                if len(pieces) != 2:
                    raise RuntimeError(f"bad merge string: {merge_value!r}")
                left, right = pieces
            elif isinstance(merge_value, list) and len(merge_value) == 2:
                left, right = str(merge_value[0]), str(merge_value[1])
            else:
                raise RuntimeError(f"bad merge entry: {merge_value!r}")

            left_bytes = left.encode("utf-8")
            right_bytes = right.encode("utf-8")

            file.write(struct.pack("<II", rank, len(left_bytes)))
            file.write(left_bytes)
            file.write(struct.pack("<I", len(right_bytes)))
            file.write(right_bytes)


def load_local_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ[key] = value


def get_hf_token() -> str | None:
    value = normalize_optional_string(os.getenv("HUGGINGFACE_HUB_TOKEN"))
    if value is not None:
        return value
    return normalize_optional_string(os.getenv("HF_TOKEN"))


def require_hf_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face streaming requires the `datasets` package. Install it with: "
            "`pip install datasets`"
        ) from exc
    return load_dataset


def normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def hf_source_string(dataset_name: str | None, subset: str | None, split: str | None) -> str:
    if dataset_name is None:
        return "<unset>"
    suffix = f"/{subset}" if subset else ""
    split_part = split or "train"
    return f"{dataset_name}{suffix}:{split_part}"


def load_hf_split(
    dataset_name: str,
    subset: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
) -> Any:
    load_dataset = require_hf_load_dataset()
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    token = get_hf_token()
    if subset is not None:
        kwargs["name"] = subset
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if token is not None:
        kwargs["token"] = token
    return load_dataset(dataset_name, **kwargs)


def maybe_shuffle_hf_dataset(dataset: Any, seed: int, buffer_size: int) -> Any:
    if buffer_size <= 0:
        return dataset
    if is_hf_iterable_dataset(dataset):
        return dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset.shuffle(seed=seed)


def is_hf_iterable_dataset(dataset: Any) -> bool:
    try:
        from datasets import IterableDataset
    except ImportError:
        return False
    return isinstance(dataset, IterableDataset)


def dataset_take(dataset: Any, count: int) -> Any:
    if count < 0:
        raise ValueError("count must be non-negative")
    if is_hf_iterable_dataset(dataset):
        return dataset.take(count)
    total = len(dataset)
    return dataset.select(range(min(count, total)))


def dataset_skip(dataset: Any, count: int) -> Any:
    if count < 0:
        raise ValueError("count must be non-negative")
    if is_hf_iterable_dataset(dataset):
        return dataset.skip(count)
    total = len(dataset)
    return dataset.select(range(min(count, total), total))


def build_hf_train_and_val_datasets(train_config: TrainConfig) -> tuple[Any, Any]:
    if train_config.hf_train_dataset is None:
        raise RuntimeError("missing --hf_train_dataset")

    train_dataset_name = train_config.hf_train_dataset
    train_subset = train_config.hf_train_subset
    train_split = train_config.hf_train_split

    val_dataset_name = train_config.hf_val_dataset or train_dataset_name
    val_subset = train_config.hf_val_subset if train_config.hf_val_subset is not None else train_subset
    val_split = train_config.hf_val_split or train_split

    same_source = (
        train_dataset_name == val_dataset_name
        and train_subset == val_subset
        and train_split == val_split
    )

    if same_source:
        val_dataset = dataset_take(
            maybe_shuffle_hf_dataset(
                load_hf_split(
                    val_dataset_name,
                    val_subset,
                    val_split,
                    train_config.hf_cache_dir,
                    train_config.hf_streaming,
                ),
                seed=train_config.seed,
                buffer_size=train_config.hf_shuffle_buffer,
            ),
            train_config.val_docs,
        )

        train_dataset = dataset_skip(
            maybe_shuffle_hf_dataset(
                load_hf_split(
                    train_dataset_name,
                    train_subset,
                    train_split,
                    train_config.hf_cache_dir,
                    train_config.hf_streaming,
                ),
                seed=train_config.seed,
                buffer_size=train_config.hf_shuffle_buffer,
            ),
            train_config.val_docs,
        )
    else:
        val_dataset = load_hf_split(
            val_dataset_name,
            val_subset,
            val_split,
            train_config.hf_cache_dir,
            train_config.hf_streaming,
        )
        val_dataset = dataset_take(val_dataset, train_config.val_docs)

        train_dataset = maybe_shuffle_hf_dataset(
            load_hf_split(
                train_dataset_name,
                train_subset,
                train_split,
                train_config.hf_cache_dir,
                train_config.hf_streaming,
            ),
            seed=train_config.seed,
            buffer_size=train_config.hf_shuffle_buffer,
        )

    if train_config.train_docs_limit is not None:
        train_dataset = dataset_take(train_dataset, train_config.train_docs_limit)

    return train_dataset, val_dataset


def build_hf_tokenizer_dataset(train_config: TrainConfig) -> Any:
    if train_config.hf_train_dataset is None:
        raise RuntimeError("missing --hf_train_dataset")

    dataset = maybe_shuffle_hf_dataset(
        load_hf_split(
            train_config.hf_train_dataset,
            train_config.hf_train_subset,
            train_config.hf_train_split,
            train_config.hf_cache_dir,
            train_config.hf_streaming,
        ),
        seed=train_config.seed,
        buffer_size=train_config.hf_shuffle_buffer,
    )
    return dataset_take(dataset, train_config.tokenizer_train_docs)


def extract_hf_text(example: Any, text_field: str) -> str | None:
    if not isinstance(example, dict):
        return None
    value = example.get(text_field)
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def normalize_sft_role(raw_role: Any) -> str:
    if not isinstance(raw_role, str):
        return USER_LABEL

    role = raw_role.strip().lower()
    if role in {"system"}:
        return SYSTEM_LABEL
    if role in {"assistant", "gpt", "bot", "model"}:
        return ASSISTANT_LABEL
    if role in {"human", "user"}:
        return USER_LABEL
    return role.capitalize() or USER_LABEL


def is_assistant_role(raw_role: Any) -> bool:
    return normalize_sft_role(raw_role) == ASSISTANT_LABEL


def extract_message_role_and_text(turn: Any) -> tuple[Any, str | None]:
    if not isinstance(turn, dict):
        return None, None

    role = turn.get("from")
    if role is None:
        role = turn.get("role")

    value = turn.get("value")
    if value is None:
        value = turn.get("content")
    if not isinstance(value, str):
        return role, None

    text = value.strip()
    if not text:
        return role, None
    return role, text


def format_sft_example_text(
    example: Any,
    conversations_field: str,
    system_prompt_field: str,
) -> str | None:
    if not isinstance(example, dict):
        return None

    chunks: list[str] = []

    system_prompt = example.get(system_prompt_field)
    if isinstance(system_prompt, str):
        system_prompt = system_prompt.strip()
        if system_prompt:
            chunks.append(f"{SYSTEM_LABEL}: {system_prompt}\n\n")

    conversations = example.get(conversations_field)
    if not isinstance(conversations, list):
        return None

    for turn in conversations:
        role, text = extract_message_role_and_text(turn)
        if text is None:
            continue
        label = normalize_sft_role(role)
        chunks.append(f"{label}: {text}\n\n")

    joined = "".join(chunks).strip()
    return joined if joined else None


def encode_text_without_specials(tokenizer: Tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False).ids


def append_encoded_piece(
    tokenizer: Tokenizer,
    token_ids: list[int],
    loss_mask: list[bool],
    text: str,
    train_on_tokens: bool,
) -> int:
    piece_ids = encode_text_without_specials(tokenizer, text)
    if not piece_ids:
        return 0
    token_ids.extend(piece_ids)
    loss_mask.extend([train_on_tokens] * len(piece_ids))
    return len(piece_ids)


def encode_hf_pretrain_example(
    example: Any,
    tokenizer: Tokenizer,
    text_field: str,
) -> EncodedSequence | None:
    text = extract_hf_text(example, text_field)
    if text is None:
        return None
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) < 2:
        return None
    return EncodedSequence(token_ids=token_ids)


def encode_hf_sft_example(
    example: Any,
    tokenizer: Tokenizer,
    bos_id: int,
    eos_id: int,
    conversations_field: str,
    system_prompt_field: str,
    train_on_prompt: bool,
) -> EncodedSequence | None:
    if not isinstance(example, dict):
        return None

    conversations = example.get(conversations_field)
    if not isinstance(conversations, list):
        return None

    token_ids: list[int] = [bos_id]
    loss_mask: list[bool] = [False]
    assistant_tokens = 0

    system_prompt = example.get(system_prompt_field)
    if isinstance(system_prompt, str):
        system_prompt = system_prompt.strip()
        if system_prompt:
            append_encoded_piece(tokenizer, token_ids, loss_mask, f"{SYSTEM_LABEL}: ", False)
            append_encoded_piece(tokenizer, token_ids, loss_mask, system_prompt, train_on_prompt)
            append_encoded_piece(tokenizer, token_ids, loss_mask, "\n\n", False)

    for turn in conversations:
        role, text = extract_message_role_and_text(turn)
        if text is None:
            continue

        role_label = normalize_sft_role(role)
        is_assistant = is_assistant_role(role)
        train_on_value = is_assistant or train_on_prompt

        append_encoded_piece(tokenizer, token_ids, loss_mask, f"{role_label}: ", False)
        assistant_tokens += append_encoded_piece(tokenizer, token_ids, loss_mask, text, train_on_value)
        append_encoded_piece(tokenizer, token_ids, loss_mask, "\n\n", False)

    token_ids.append(eos_id)
    loss_mask.append(False)

    if len(token_ids) < 2:
        return None
    if not train_on_prompt and assistant_tokens == 0:
        return None

    return EncodedSequence(token_ids=token_ids, loss_mask=loss_mask)


def filter_sft_example_by_length(
    encoded: EncodedSequence | None,
    min_example_tokens: int,
    max_example_tokens: int | None,
) -> EncodedSequence | None:
    if encoded is None:
        return None

    input_tokens = len(encoded.token_ids) - 1
    if input_tokens < 1:
        return None
    if input_tokens < min_example_tokens:
        return None
    if max_example_tokens is not None and input_tokens > max_example_tokens:
        return None
    return encoded


def iter_hf_tokenizer_texts(train_config: TrainConfig) -> Iterator[str]:
    dataset = build_hf_tokenizer_dataset(train_config)

    if train_config.task == "pretrain":
        for example in dataset:
            text = extract_hf_text(example, train_config.hf_text_field)
            if text is not None:
                yield text
        return

    for example in dataset:
        text = format_sft_example_text(
            example,
            conversations_field=train_config.hf_conversations_field,
            system_prompt_field=train_config.hf_system_prompt_field,
        )
        if text is not None:
            yield text


def sample_random_batch(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor | None,
    ctx_len: int,
    batch_size: int,
    device: torch.device,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - ctx_len - 1
    if max_start < 0:
        raise ValueError("token buffer is too small for ctx_len")

    for _ in range(32):
        starts = torch.randint(0, max_start + 1, (batch_size,), dtype=torch.long)
        starts_list = starts.tolist()

        x_cpu = torch.stack([tokens[start:start + ctx_len] for start in starts_list])
        y_cpu = torch.stack([tokens[start + 1:start + 1 + ctx_len] for start in starts_list])

        if loss_mask is None:
            return x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)

        y_mask_cpu = torch.stack([loss_mask[start + 1:start + 1 + ctx_len] for start in starts_list])
        if not bool(y_mask_cpu.any()):
            continue

        y_cpu = y_cpu.masked_fill(~y_mask_cpu, pad_id)
        return x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)

    raise RuntimeError("failed to sample a batch containing any supervised target tokens")


class RandomSpanLoader:
    def __init__(
        self,
        tokens: torch.Tensor,
        ctx_len: int,
        batch_size: int,
        device: torch.device,
        pad_id: int,
        loss_mask: torch.Tensor | None = None,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D tensor")
        if tokens.numel() < ctx_len + 2:
            raise ValueError("token file is too small for ctx_len")
        if loss_mask is not None:
            if loss_mask.ndim != 1:
                raise ValueError("loss_mask must be a 1D tensor")
            if loss_mask.numel() != tokens.numel():
                raise ValueError("loss_mask must have the same length as tokens")

        self.tokens = tokens
        self.loss_mask = loss_mask
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.device = device
        self.pad_id = pad_id

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_random_batch(
            self.tokens,
            self.loss_mask,
            self.ctx_len,
            self.batch_size,
            self.device,
            self.pad_id,
        )


def encoded_sequence_to_padded_example_tensors(
    encoded: EncodedSequence,
    ctx_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if encoded.loss_mask is None:
        raise RuntimeError("SFT example batching requires a loss mask")
    if len(encoded.loss_mask) != len(encoded.token_ids):
        raise RuntimeError("loss_mask length did not match token_ids length")

    input_ids = encoded.token_ids[:-1]
    target_ids = encoded.token_ids[1:]
    target_mask = encoded.loss_mask[1:]

    if len(input_ids) > ctx_len:
        input_ids = input_ids[:ctx_len]
        target_ids = target_ids[:ctx_len]
        target_mask = target_mask[:ctx_len]

    pad_tokens = ctx_len - len(input_ids)
    x_values = input_ids + [pad_id] * pad_tokens
    y_values = target_ids + [pad_id] * pad_tokens
    y_mask_values = target_mask + [False] * pad_tokens

    x = torch.tensor(x_values, dtype=torch.long)
    y = torch.tensor(y_values, dtype=torch.long)
    y_mask = torch.tensor(y_mask_values, dtype=torch.bool)
    y = y.masked_fill(~y_mask, pad_id)
    return x, y


class RollingSpanLoader:
    def __init__(
        self,
        example_iterable: Any,
        encoder: Callable[[Any], EncodedSequence | None],
        ctx_len: int,
        batch_size: int,
        device: torch.device,
        pad_id: int,
        buffer_tokens: int,
        refresh_buffer_tokens: int,
        uses_loss_mask: bool,
    ) -> None:
        if buffer_tokens < ctx_len + 2:
            raise ValueError("train_buffer_tokens must be at least ctx_len + 2")
        if refresh_buffer_tokens <= 0:
            raise ValueError("refresh_buffer_tokens must be positive")

        self.example_iterable = example_iterable
        self.encoder = encoder
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.device = device
        self.pad_id = pad_id
        self.buffer_tokens = buffer_tokens
        self.refresh_buffer_tokens = refresh_buffer_tokens
        self.uses_loss_mask = uses_loss_mask

        self.epoch = 0
        if hasattr(self.example_iterable, "set_epoch"):
            self.example_iterable.set_epoch(self.epoch)
        self.iterator = iter(self.example_iterable)

        self.tokens = torch.empty(0, dtype=torch.long)
        self.loss_mask = torch.empty(0, dtype=torch.bool) if uses_loss_mask else None
        self.sampled_tokens_since_refresh = 0
        self.examples_loaded = 0

        self._append_until(self.buffer_tokens)

    def _reset_iterator(self) -> None:
        self.epoch += 1
        if hasattr(self.example_iterable, "set_epoch"):
            self.example_iterable.set_epoch(self.epoch)
        self.iterator = iter(self.example_iterable)

    def _next_encoded_sequence(self) -> EncodedSequence:
        while True:
            try:
                example = next(self.iterator)
            except StopIteration:
                self._reset_iterator()
                continue

            encoded = self.encoder(example)
            if encoded is None:
                continue
            if len(encoded.token_ids) < 2:
                continue
            if self.uses_loss_mask:
                if encoded.loss_mask is None:
                    raise RuntimeError("SFT loader expected a loss mask")
                if len(encoded.loss_mask) != len(encoded.token_ids):
                    raise RuntimeError("loss_mask length did not match token_ids length")
            return encoded

    def _append_until(self, minimum_new_tokens: int) -> None:
        token_chunks: list[torch.Tensor] = []
        loss_mask_chunks: list[torch.Tensor] = []
        added_tokens = 0

        while added_tokens < minimum_new_tokens or self.tokens.numel() + added_tokens < self.ctx_len + 2:
            encoded = self._next_encoded_sequence()
            token_chunks.append(torch.tensor(encoded.token_ids, dtype=torch.long))
            if self.uses_loss_mask:
                assert encoded.loss_mask is not None
                loss_mask_chunks.append(torch.tensor(encoded.loss_mask, dtype=torch.bool))
            added_tokens += len(encoded.token_ids)
            self.examples_loaded += 1

        new_tokens = torch.cat(token_chunks)
        self.tokens = torch.cat((self.tokens, new_tokens))

        if self.uses_loss_mask:
            assert self.loss_mask is not None
            self.loss_mask = torch.cat((self.loss_mask, torch.cat(loss_mask_chunks)))

        if self.tokens.numel() > self.buffer_tokens:
            self.tokens = self.tokens[-self.buffer_tokens:]
            if self.loss_mask is not None:
                self.loss_mask = self.loss_mask[-self.buffer_tokens:]

    def _refresh(self) -> None:
        if self.tokens.numel() > self.refresh_buffer_tokens + self.ctx_len + 1:
            self.tokens = self.tokens[self.refresh_buffer_tokens:]
            if self.loss_mask is not None:
                self.loss_mask = self.loss_mask[self.refresh_buffer_tokens:]
        self._append_until(self.refresh_buffer_tokens)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sampled_tokens_since_refresh >= self.refresh_buffer_tokens:
            self._refresh()
            self.sampled_tokens_since_refresh = 0

        batch = sample_random_batch(
            self.tokens,
            self.loss_mask,
            self.ctx_len,
            self.batch_size,
            self.device,
            self.pad_id,
        )
        self.sampled_tokens_since_refresh += self.batch_size * self.ctx_len
        return batch


class RandomExampleLoader:
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("inputs and targets must be 2D tensors")
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have the same shape")
        if inputs.shape[0] < 1:
            raise ValueError("example loader needs at least one example")

        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.device = device

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.inputs.size(0), (self.batch_size,), dtype=torch.long)
        x_cpu = self.inputs[indices]
        y_cpu = self.targets[indices]
        return x_cpu.to(self.device, non_blocking=True), y_cpu.to(self.device, non_blocking=True)


class RollingExampleLoader:
    def __init__(
        self,
        example_iterable: Any,
        encoder: Callable[[Any], EncodedSequence | None],
        ctx_len: int,
        batch_size: int,
        device: torch.device,
        pad_id: int,
        buffer_tokens: int,
        refresh_buffer_tokens: int,
    ) -> None:
        if refresh_buffer_tokens <= 0:
            raise ValueError("refresh_buffer_tokens must be positive")

        self.example_iterable = example_iterable
        self.encoder = encoder
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.device = device
        self.pad_id = pad_id
        self.buffer_examples = max(batch_size, buffer_tokens // max(1, ctx_len))
        self.refresh_examples = max(batch_size, refresh_buffer_tokens // max(1, ctx_len))

        self.epoch = 0
        if hasattr(self.example_iterable, "set_epoch"):
            self.example_iterable.set_epoch(self.epoch)
        self.iterator = iter(self.example_iterable)

        self.inputs = torch.empty((0, ctx_len), dtype=torch.long)
        self.targets = torch.empty((0, ctx_len), dtype=torch.long)
        self.sampled_examples_since_refresh = 0
        self.examples_loaded = 0

        self._append_until(self.buffer_examples)

    def _reset_iterator(self) -> None:
        self.epoch += 1
        if hasattr(self.example_iterable, "set_epoch"):
            self.example_iterable.set_epoch(self.epoch)
        self.iterator = iter(self.example_iterable)

    def _next_encoded_sequence(self) -> EncodedSequence:
        while True:
            try:
                example = next(self.iterator)
            except StopIteration:
                self._reset_iterator()
                continue

            encoded = self.encoder(example)
            if encoded is None:
                continue
            return encoded

    def _append_until(self, minimum_new_examples: int) -> None:
        input_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []

        while len(input_chunks) < minimum_new_examples or self.inputs.size(0) + len(input_chunks) < self.batch_size:
            encoded = self._next_encoded_sequence()
            x, y = encoded_sequence_to_padded_example_tensors(encoded, self.ctx_len, self.pad_id)
            input_chunks.append(x)
            target_chunks.append(y)
            self.examples_loaded += 1

        new_inputs = torch.stack(input_chunks)
        new_targets = torch.stack(target_chunks)

        self.inputs = torch.cat((self.inputs, new_inputs))
        self.targets = torch.cat((self.targets, new_targets))

        if self.inputs.size(0) > self.buffer_examples:
            self.inputs = self.inputs[-self.buffer_examples:]
            self.targets = self.targets[-self.buffer_examples:]

    def _refresh(self) -> None:
        if self.inputs.size(0) > self.refresh_examples:
            self.inputs = self.inputs[self.refresh_examples:]
            self.targets = self.targets[self.refresh_examples:]
        self._append_until(self.refresh_examples)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sampled_examples_since_refresh >= self.refresh_examples:
            self._refresh()
            self.sampled_examples_since_refresh = 0

        indices = torch.randint(0, self.inputs.size(0), (self.batch_size,), dtype=torch.long)
        x_cpu = self.inputs[indices]
        y_cpu = self.targets[indices]
        self.sampled_examples_since_refresh += self.batch_size
        return x_cpu.to(self.device, non_blocking=True), y_cpu.to(self.device, non_blocking=True)


def encode_iterable_to_tensors(
    examples: Iterable[Any],
    encoder: Callable[[Any], EncodedSequence | None],
    uses_loss_mask: bool,
    ctx_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    token_chunks: list[torch.Tensor] = []
    loss_mask_chunks: list[torch.Tensor] = []
    docs_used = 0

    for example in examples:
        encoded = encoder(example)
        if encoded is None:
            continue
        if len(encoded.token_ids) < 2:
            continue
        if uses_loss_mask:
            if encoded.loss_mask is None:
                raise RuntimeError("SFT validation expected a loss mask")
            if len(encoded.loss_mask) != len(encoded.token_ids):
                raise RuntimeError("loss_mask length did not match token_ids length")
            loss_mask_chunks.append(torch.tensor(encoded.loss_mask, dtype=torch.bool))
        token_chunks.append(torch.tensor(encoded.token_ids, dtype=torch.long))
        docs_used += 1

    if not token_chunks:
        raise RuntimeError("no usable examples were found for the requested dataset split")

    tokens = torch.cat(token_chunks)
    if tokens.numel() < ctx_len + 2:
        raise RuntimeError("validation token set is too small for the requested ctx_len")

    if not uses_loss_mask:
        return tokens, None, docs_used

    return tokens, torch.cat(loss_mask_chunks), docs_used


def encode_iterable_to_padded_examples(
    examples: Iterable[Any],
    encoder: Callable[[Any], EncodedSequence | None],
    ctx_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    input_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    docs_used = 0

    for example in examples:
        encoded = encoder(example)
        if encoded is None:
            continue
        x, y = encoded_sequence_to_padded_example_tensors(encoded, ctx_len, pad_id)
        input_chunks.append(x)
        target_chunks.append(y)
        docs_used += 1

    if not input_chunks:
        raise RuntimeError("no usable SFT examples were found for the requested dataset split")

    return torch.stack(input_chunks), torch.stack(target_chunks), docs_used


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalDepthwiseConvK3(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, 3))
        self.bias = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        #
        # Causal depthwise k=3:
        #   y[t,c] = x[t,c]*w[c,2] + x[t-1,c]*w[c,1] + x[t-2,c]*w[c,0] + b[c]
        #
        # Explicit form makes the later C implementation obvious.
        x0 = x
        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        x2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))

        w_current = self.weight[:, 2]
        w_prev1 = self.weight[:, 1]
        w_prev2 = self.weight[:, 0]

        return x0 * w_current + x1 * w_prev1 + x2 * w_prev2 + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.ctx_len = config.ctx_len

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.ctx_len, config.ctx_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, -1e4)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        return self.o_proj(out)


class ReLUMlp(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        hidden_dim = config.d_model * config.mlp_mult

        self.up = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.up(x))
        x = self.down(x)
        return self.dropout(x)


class SharedTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = ReLUMlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LoopedTransformerLm(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.ctx_len, config.d_model)

        self.embed_conv = CausalDepthwiseConvK3(config.d_model)
        self.blocks = nn.ModuleList(
            SharedTransformerBlock(config) for _ in range(config.unique_blocks)
        )
        self.final_norm = RMSNorm(config.d_model)

        # Tied output head uses token_emb.weight.T. Bias is separate.
        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape

        if seq_len > self.config.ctx_len:
            raise ValueError("sequence length exceeds ctx_len")

        positions = torch.arange(0, seq_len, device=idx.device, dtype=torch.long)

        x = self.token_emb(idx)
        x = x + self.pos_emb(positions)[None, :, :]

        x = self.embed_conv(x)
        x = self.dropout(x)

        for block in self.blocks:
            for _ in range(self.config.loops_per_pass):
                x = block(x)

        x = self.final_norm(x)

        logits = torch.matmul(x, self.token_emb.weight.t()) + self.lm_head_bias

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int = 40,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.ctx_len:]
            logits, _ = self(idx_cond, None)

            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                filtered = torch.full_like(logits, -float("inf"))
                filtered.scatter_(dim=-1, index=indices, src=values)
                logits = filtered

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if eos_id is not None and int(next_token[0, 0].item()) == eos_id:
                break

        return idx


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_autocast_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32

    raise ValueError("dtype must be fp16, bf16, or fp32")


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)

    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: LoopedTransformerLm,
    loader: Any,
    batches: int,
    autocast_dtype: torch.dtype,
) -> float:
    model.eval()
    losses: list[float] = []

    for _ in range(batches):
        x, y = loader.next_batch()

        if x.device.type == "cuda" and autocast_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("evaluation loss was None")

        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    path: str,
    model: LoopedTransformerLm,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    val_loss: float,
) -> None:
    ensure_parent_dir(path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_init_checkpoint(path: str, model: LoopedTransformerLm) -> tuple[int | None, float | None]:
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError(f"checkpoint {path!r} does not contain `model_state`")

    model.load_state_dict(model_state)

    step = checkpoint.get("step")
    if not isinstance(step, int):
        step = None

    val_loss = checkpoint.get("val_loss")
    if isinstance(val_loss, (int, float)):
        return step, float(val_loss)

    return step, None


def sample_text(
    model: LoopedTransformerLm,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    encoded = tokenizer.encode(prompt)
    ids = encoded.ids

    if len(ids) == 0:
        ids = [model.config.bos_id]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_k=40,
        eos_id=model.config.eos_id,
    )

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def build_hf_loaders(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    device: torch.device,
) -> tuple[Any, Any]:
    train_dataset, val_dataset = build_hf_train_and_val_datasets(train_config)

    uses_loss_mask = train_config.task == "sft"
    if uses_loss_mask:
        effective_max_tokens = train_config.sft_max_example_tokens

        def encoder(example: Any) -> EncodedSequence | None:
            raw = encode_hf_sft_example(
                example,
                tokenizer,
                bos_id=model_config.bos_id,
                eos_id=model_config.eos_id,
                conversations_field=train_config.hf_conversations_field,
                system_prompt_field=train_config.hf_system_prompt_field,
                train_on_prompt=train_config.sft_train_on_prompt,
            )
            return filter_sft_example_by_length(
                raw,
                min_example_tokens=train_config.sft_min_example_tokens,
                max_example_tokens=effective_max_tokens,
            )
    else:
        encoder = lambda example: encode_hf_pretrain_example(
            example,
            tokenizer,
            text_field=train_config.hf_text_field,
        )

    if uses_loss_mask:
        val_inputs, val_targets, val_docs_used = encode_iterable_to_padded_examples(
            val_dataset,
            encoder=encoder,
            ctx_len=model_config.ctx_len,
            pad_id=model_config.pad_id,
        )
        supervised_targets = int((val_targets != model_config.pad_id).sum().item())
        print(f"val docs: {val_docs_used:,}")
        print(
            f"val examples: {val_inputs.size(0):,} | "
            f"supervised targets: {supervised_targets:,}"
        )

        train_loader = RollingExampleLoader(
            example_iterable=train_dataset,
            encoder=encoder,
            ctx_len=model_config.ctx_len,
            batch_size=train_config.batch_size,
            device=device,
            pad_id=model_config.pad_id,
            buffer_tokens=train_config.train_buffer_tokens,
            refresh_buffer_tokens=train_config.refresh_buffer_tokens,
        )
        val_loader = RandomExampleLoader(
            val_inputs,
            val_targets,
            train_config.batch_size,
            device,
        )

        if effective_max_tokens is None:
            print(f"sft overlength handling: truncate examples to ctx_len={model_config.ctx_len}")
        else:
            print(
                f"sft example length filter: "
                f"{train_config.sft_min_example_tokens}..{effective_max_tokens} tokens"
            )
        print(f"train example buffer: {train_loader.inputs.size(0):,}")
    else:
        val_tokens, val_loss_mask, val_docs_used = encode_iterable_to_tensors(
            val_dataset,
            encoder=encoder,
            uses_loss_mask=uses_loss_mask,
            ctx_len=model_config.ctx_len,
        )
        print(f"val docs: {val_docs_used:,}")
        print(f"val tokens: {val_tokens.numel():,}")

        train_loader = RollingSpanLoader(
            example_iterable=train_dataset,
            encoder=encoder,
            ctx_len=model_config.ctx_len,
            batch_size=train_config.batch_size,
            device=device,
            pad_id=model_config.pad_id,
            buffer_tokens=train_config.train_buffer_tokens,
            refresh_buffer_tokens=train_config.refresh_buffer_tokens,
            uses_loss_mask=uses_loss_mask,
        )
        val_loader = RandomSpanLoader(
            val_tokens,
            model_config.ctx_len,
            train_config.batch_size,
            device,
            model_config.pad_id,
            loss_mask=val_loss_mask,
        )

        print(f"train token buffer: {train_loader.tokens.numel():,}")
    print(f"local storage budget: {train_config.local_storage_budget_gb:.1f} GB max")
    if train_config.hf_streaming:
        print(
            "dataset storage mode: streaming from Hugging Face "
            "(full dataset download avoided)"
        )
    else:
        print(
            "dataset storage mode: map-style Hugging Face download "
            "(dataset cached locally before training)"
        )

    return train_loader, val_loader


def parse_args() -> tuple[argparse.Namespace, TrainConfig]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_source", choices=["text", "huggingface"], default="text")
    parser.add_argument("--task", choices=["pretrain", "sft"], default="pretrain")

    parser.add_argument("--train_text", default=None)
    parser.add_argument("--val_text", default=None)
    parser.add_argument("--out", required=True)

    parser.add_argument("--hf_train_dataset", default=None)
    parser.add_argument("--hf_train_subset", default=None)
    parser.add_argument("--hf_train_split", default="train")
    parser.add_argument("--hf_val_dataset", default=None)
    parser.add_argument("--hf_val_subset", default=None)
    parser.add_argument("--hf_val_split", default=None)
    parser.add_argument("--hf_no_streaming", action="store_true")
    parser.add_argument("--hf_text_field", default="text")
    parser.add_argument("--hf_conversations_field", default="conversations")
    parser.add_argument("--hf_system_prompt_field", default="system_prompt")
    parser.add_argument("--hf_shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--tokenizer_train_docs", type=int, default=50_000)
    parser.add_argument("--train_docs_limit", type=int, default=None)
    parser.add_argument("--val_docs", type=int, default=2_000)
    parser.add_argument("--train_buffer_tokens", type=int, default=1_000_000)
    parser.add_argument("--refresh_buffer_tokens", type=int, default=250_000)
    parser.add_argument("--sft_train_on_prompt", action="store_true")
    parser.add_argument("--sft_min_example_tokens", type=int, default=0)
    parser.add_argument("--sft_max_example_tokens", type=int, default=0)
    parser.add_argument("--hf_cache_dir", default=".cache/huggingface")
    parser.add_argument("--local_storage_budget_gb", type=float, default=50.0)

    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer_out", default="tokenizer_qwends.json")
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--export_tokenizer_cbin", default=None)
    parser.add_argument("--init_checkpoint", default=None)

    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--unique_blocks", type=int, default=3)
    parser.add_argument("--loops_per_pass", type=int, default=6)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--sample_prompt", default="The answer is")
    parser.add_argument("--sample_tokens", type=int, default=120)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    args.train_text = normalize_optional_string(args.train_text)
    args.val_text = normalize_optional_string(args.val_text)
    args.hf_train_dataset = normalize_optional_string(args.hf_train_dataset)
    args.hf_train_subset = normalize_optional_string(args.hf_train_subset)
    args.hf_train_split = normalize_optional_string(args.hf_train_split) or "train"
    args.hf_val_dataset = normalize_optional_string(args.hf_val_dataset)
    args.hf_val_subset = normalize_optional_string(args.hf_val_subset)
    args.hf_val_split = normalize_optional_string(args.hf_val_split)
    args.hf_cache_dir = normalize_optional_string(args.hf_cache_dir) or ".cache/huggingface"

    if args.data_source == "text":
        if args.train_text is None or args.val_text is None:
            parser.error("--data_source text requires both --train_text and --val_text")
        if args.task != "pretrain":
            parser.error("--data_source text currently only supports --task pretrain")
    else:
        if args.hf_train_dataset is None:
            parser.error("--data_source huggingface requires --hf_train_dataset")
        if args.val_docs < 1:
            parser.error("--val_docs must be at least 1")
        if args.tokenizer_train_docs < 1:
            parser.error("--tokenizer_train_docs must be at least 1")
        if args.train_buffer_tokens < args.ctx_len + 2:
            parser.error("--train_buffer_tokens must be at least ctx_len + 2")
        if args.refresh_buffer_tokens < 1:
            parser.error("--refresh_buffer_tokens must be at least 1")
    if args.sft_min_example_tokens < 0:
        parser.error("--sft_min_example_tokens must be non-negative")
    if args.sft_max_example_tokens < 0:
        parser.error("--sft_max_example_tokens must be non-negative")
    if args.sft_max_example_tokens > 0 and args.sft_min_example_tokens > args.sft_max_example_tokens:
        parser.error("--sft_min_example_tokens cannot exceed --sft_max_example_tokens")
    if args.unique_blocks < 1:
        parser.error("--unique_blocks must be at least 1")
    if args.loops_per_pass < 1:
        parser.error("--loops_per_pass must be at least 1")

    if args.tokenizer is None and not args.train_tokenizer:
        parser.error("pass --train_tokenizer or provide --tokenizer tokenizer.json")

    train_config = TrainConfig(
        out=args.out,
        tokenizer=args.tokenizer,
        tokenizer_out=args.tokenizer_out,
        train_tokenizer=args.train_tokenizer,
        export_tokenizer_cbin=args.export_tokenizer_cbin,
        init_checkpoint=args.init_checkpoint,
        data_source=args.data_source,
        task=args.task,
        train_text=args.train_text,
        val_text=args.val_text,
        hf_train_dataset=args.hf_train_dataset,
        hf_train_subset=args.hf_train_subset,
        hf_train_split=args.hf_train_split,
        hf_val_dataset=args.hf_val_dataset,
        hf_val_subset=args.hf_val_subset,
        hf_val_split=args.hf_val_split,
        hf_streaming=not args.hf_no_streaming,
        hf_text_field=args.hf_text_field,
        hf_conversations_field=args.hf_conversations_field,
        hf_system_prompt_field=args.hf_system_prompt_field,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
        tokenizer_train_docs=args.tokenizer_train_docs,
        train_docs_limit=args.train_docs_limit,
        val_docs=args.val_docs,
        train_buffer_tokens=args.train_buffer_tokens,
        refresh_buffer_tokens=args.refresh_buffer_tokens,
        sft_train_on_prompt=args.sft_train_on_prompt,
        sft_min_example_tokens=args.sft_min_example_tokens,
        sft_max_example_tokens=(args.sft_max_example_tokens or None),
        hf_cache_dir=args.hf_cache_dir,
        local_storage_budget_gb=args.local_storage_budget_gb,
        vocab_size=args.vocab_size,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
    )

    return args, train_config


def main() -> None:
    args, train_config = parse_args()
    load_local_env_file(".env")

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = get_device(train_config.device)
    autocast_dtype = get_autocast_dtype(train_config.dtype)

    if train_config.train_tokenizer:
        if train_config.data_source == "text":
            assert train_config.train_text is not None
            tokenizer = train_bpe_tokenizer_from_file(
                train_config.train_text,
                train_config.tokenizer_out,
                train_config.vocab_size,
            )
        else:
            tokenizer = train_bpe_tokenizer_from_iterator(
                iter_hf_tokenizer_texts(train_config),
                train_config.tokenizer_out,
                train_config.vocab_size,
                length=train_config.tokenizer_train_docs,
            )
        tokenizer_path = train_config.tokenizer_out
    else:
        if train_config.tokenizer is None:
            raise ValueError("pass --train_tokenizer or provide --tokenizer tokenizer.json")
        tokenizer = load_bpe_tokenizer(train_config.tokenizer)
        tokenizer_path = train_config.tokenizer

    if train_config.export_tokenizer_cbin is not None:
        export_tokenizer_cbin(tokenizer_path, train_config.export_tokenizer_cbin)
        print(f"exported tokenizer cbin: {train_config.export_tokenizer_cbin}")

    bos_id = special_id(tokenizer, BOS_TEXT)
    eos_id = special_id(tokenizer, EOS_TEXT)
    pad_id = special_id(tokenizer, PAD_TEXT)
    unk_id = special_id(tokenizer, UNK_TEXT)

    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        ctx_len=args.ctx_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        unique_blocks=args.unique_blocks,
        loops_per_pass=args.loops_per_pass,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
    )

    print(f"device: {device}")
    print(f"tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print(f"special ids: bos={bos_id} eos={eos_id} pad={pad_id} unk={unk_id}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if train_config.data_source == "text":
        assert train_config.train_text is not None
        assert train_config.val_text is not None
        train_tokens = load_tokens(train_config.train_text, tokenizer)
        val_tokens = load_tokens(train_config.val_text, tokenizer)

        print(f"train tokens: {train_tokens.numel():,}")
        print(f"val tokens:   {val_tokens.numel():,}")

        train_loader: RandomSpanLoader | RollingSpanLoader = RandomSpanLoader(
            train_tokens,
            model_config.ctx_len,
            train_config.batch_size,
            device,
            model_config.pad_id,
        )
        val_loader = RandomSpanLoader(
            val_tokens,
            model_config.ctx_len,
            train_config.batch_size,
            device,
            model_config.pad_id,
        )
    else:
        print(
            "hf train source: "
            f"{hf_source_string(train_config.hf_train_dataset, train_config.hf_train_subset, train_config.hf_train_split)}"
        )
        print(
            "hf val source:   "
            f"{hf_source_string(train_config.hf_val_dataset or train_config.hf_train_dataset, train_config.hf_val_subset if train_config.hf_val_subset is not None else train_config.hf_train_subset, train_config.hf_val_split or train_config.hf_train_split)}"
        )
        print(f"huggingface auth token: {'loaded' if get_hf_token() is not None else 'not found'}")
        train_loader, val_loader = build_hf_loaders(train_config, tokenizer, model_config, device)

    model = LoopedTransformerLm(model_config).to(device)

    if train_config.init_checkpoint is not None:
        loaded_step, loaded_val_loss = load_init_checkpoint(train_config.init_checkpoint, model)
        summary = f"loaded init checkpoint: {train_config.init_checkpoint}"
        if loaded_step is not None:
            summary += f" (saved_step={loaded_step}"
            if loaded_val_loss is not None:
                summary += f", val_loss={loaded_val_loss:.4f}"
            summary += ")"
        print(summary)

    param_count = count_parameters(model)
    print(f"parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_config.weight_decay,
    )

    best_val = float("inf")
    start_time = time.time()

    model.train()

    for step in range(train_config.steps):
        lr = cosine_lr(step, train_config.lr, train_config.warmup_steps, train_config.steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_loader.next_batch()

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and autocast_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("training loss was None")

        loss.backward()

        # Shared-loop models can spike gradients early.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            tokens_seen = (step + 1) * train_config.batch_size * model_config.ctx_len
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)

            print(
                f"step {step:6d} | "
                f"loss {loss.item():.4f} | "
                f"ppl {math.exp(min(loss.item(), 20.0)):.2f} | "
                f"lr {lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )

        if step > 0 and step % train_config.eval_every == 0:
            val_loss = evaluate(model, val_loader, train_config.eval_batches, autocast_dtype)
            val_ppl = math.exp(min(val_loss, 20.0))

            print(
                f"eval step {step:6d} | "
                f"val_loss {val_loss:.4f} | "
                f"val_ppl {val_ppl:.2f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_path = train_config.out.replace(".pt", ".best.pt")
                save_checkpoint(best_path, model, optimizer, model_config, train_config, step, val_loss)
                print(f"saved best: {best_path}")

        if step > 0 and step % train_config.sample_every == 0:
            text = sample_text(
                model,
                tokenizer,
                train_config.sample_prompt,
                train_config.sample_tokens,
                device,
            )
            print("sample:")
            print(text)
            print("---")

        if step > 0 and step % train_config.save_every == 0:
            save_checkpoint(train_config.out, model, optimizer, model_config, train_config, step, best_val)
            print(f"saved: {train_config.out}")

    final_val = evaluate(model, val_loader, train_config.eval_batches, autocast_dtype)
    save_checkpoint(train_config.out, model, optimizer, model_config, train_config, train_config.steps, final_val)

    print(f"final val_loss {final_val:.4f} | final_ppl {math.exp(min(final_val, 20.0)):.2f}")
    print(f"saved final: {train_config.out}")


if __name__ == "__main__":
    main()
