#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from train import (
    add_chat_special_tokens,
    encode_text_without_specials,
    load_bpe_tokenizer,
    load_local_env_file,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
REJECT_ASSISTANT_PHRASES = (
    "as an ai",
    "personal feelings",
    "cannot access",
    "i am a model",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} was not a JSON object")
            examples.append(value)
    if not examples:
        raise RuntimeError(f"{path} did not contain any examples")
    return examples


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return load_jsonl(path)


def usable_seed_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [example for example in examples if clean_chat_messages(example) is not None]


def message_pair(example: dict[str, Any]) -> tuple[str, str] | None:
    messages = clean_chat_messages(example)
    if messages is None or len(messages) != 2:
        return None
    return messages[0]["content"], messages[1]["content"]


def clean_chat_messages(example: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if len(messages) % 2 != 0:
        return None

    clean: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            return None
        text = message.get("content")
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        clean.append({"role": expected_role, "content": text})
    return clean


def message_key(example: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    messages = clean_chat_messages(example)
    if messages is None:
        return None
    return tuple((message["role"], message["content"]) for message in messages)


def compact_messages_len(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return 1 + len(messages) + sum(
        len(encode_text_without_specials(tokenizer, message["content"]))
        for message in messages
    )


def assistant_len(tokenizer: Any, assistant_text: str) -> int:
    return len(encode_text_without_specials(tokenizer, assistant_text))


def build_prompt(
    seed_examples: list[dict[str, Any]],
    batch_size: int,
    multi_turn_ratio: float,
    min_multi_turns: int,
    max_turns: int,
) -> str:
    shots: list[str] = []
    for index, example in enumerate(seed_examples, start=1):
        messages = clean_chat_messages(example)
        if messages is None:
            continue
        lines = [f"Example {index}"]
        for message in messages:
            label = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{label}: {message['content']}")
        shots.append("\n".join(lines))

    multi_turn_count = round(batch_size * multi_turn_ratio)
    single_turn_count = batch_size - multi_turn_count

    return (
        "Create short, useful chat training conversations for a tiny handheld toy language model with a 128-token context.\n"
        "The assistant should sound like a helpful chat buddy, not like an AI policy system.\n"
        "Target general conversation and fact Q&A: greetings, preferences, jokes, trivia, definitions, simple science, "
        "history, geography, common knowledge, quick explanations, translations, recipes, and one-step everyday advice.\n"
        "Each assistant answer must be concise, direct, complete, and usually one sentence or a short phrase.\n"
        "Prefer compressed answers like 'Paris.', 'Use fresh beans.', 'Try cinnamon near the entry point.'.\n"
        f"Generate exactly {batch_size} examples: {single_turn_count} single-turn and {multi_turn_count} multi-turn.\n"
        f"Multi-turn examples must have {min_multi_turns} to {max_turns} user turns, with one assistant reply after each user turn.\n"
        "Use a spread of lengths: many 2-4 turn chats, some 5-7 turn chats, and a few 8-10 turn chats.\n"
        "Longer chats must use tiny turns: brief preferences, yes/no answers, small corrections, trivia games, or quick follow-ups.\n"
        "Multi-turn conversations should maintain context with natural follow-ups, corrections, or brief clarifications.\n"
        "For preference questions, answer with a plausible preference instead of disclaiming feelings or identity.\n"
        "Avoid current time, current weather, live news, stock prices, device state, or location-specific live facts.\n"
        "Do not use phrases like \"as an AI\", \"I don't have personal feelings\", \"I cannot access\", or \"I am a model\".\n"
        "Use only ASCII characters in user and assistant text.\n"
        "Avoid medical/legal/financial advice beyond harmless general tips, tool calls, code blocks, markdown tables, "
        "XML, JSON inside message text, chain-of-thought, and references "
        "to being a model.\n"
        "Return only JSON with an \"examples\" array. Each item must have this shape, with more alternating turns allowed:\n"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"...\"}]}\n\n"
        "Style examples:\n"
        + "\n\n".join(shots)
        + f"\n\nGenerate exactly {batch_size} new diverse examples now, no extras."
    )


def parse_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            try:
                value = json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                salvaged = salvage_json_objects(stripped)
                if salvaged:
                    return salvaged
                raise
        else:
            salvaged = salvage_json_objects(stripped)
            if salvaged:
                return salvaged
            raise RuntimeError("model response did not contain a JSON array")
    if isinstance(value, dict):
        for key in ("examples", "items", "data", "pairs"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    if isinstance(value, list):
        return value
    raise RuntimeError("model response JSON did not contain an examples array")


def salvage_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict) and isinstance(value.get("messages"), list):
            values.append(value)
        index = start + max(end, 1)
    return values


def short_error(exc: BaseException, max_chars: int = 240) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = body.replace("\n", " ").strip()
    if body:
        return f"HTTP Error {exc.code}: {exc.reason}: {body}"
    return f"HTTP Error {exc.code}: {exc.reason}"


def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    response_format: bool,
    exclude_reasoning: bool,
    reasoning_effort: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate compact supervised fine-tuning chat data. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    if exclude_reasoning or reasoning_effort:
        reasoning: dict[str, Any] = {}
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort
        if exclude_reasoning:
            reasoning["exclude"] = True
        payload["reasoning"] = reasoning
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/GPT3DS",
            "X-Title": "GPT3DS SFT synthesis",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response had no choices; keys={sorted(body.keys())}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"OpenRouter choice had no message; keys={sorted(choices[0].keys())}")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"OpenRouter message had no text content; keys={sorted(message.keys())}")
    return content


def request_values(
    attempt: int,
    api_key: str,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[int, list[Any] | None, str | None]:
    try:
        content = call_openrouter(
            api_key,
            args.model,
            prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            response_format=not args.no_response_format,
            exclude_reasoning=args.exclude_reasoning,
            reasoning_effort=args.reasoning_effort,
        )
        return attempt, parse_json_array(content), None
    except urllib.error.HTTPError as exc:
        return attempt, None, short_error(http_error_message(exc))
    except (RuntimeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return attempt, None, short_error(exc)


def validate_example(
    value: Any,
    tokenizer: Any,
    max_pair_input_tokens: int,
    max_assistant_tokens: int,
    max_turns: int,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    messages = clean_chat_messages(value)
    if messages is None:
        return None

    assistant_count = sum(1 for message in messages if message["role"] == "assistant")
    if assistant_count < 1 or assistant_count > max_turns:
        return None
    if len(messages) != assistant_count * 2:
        return None

    assistant_tokens_total = 0
    for index, message in enumerate(messages):
        text = message["content"]
        if not text.isascii() or "\n" in text:
            return None
        if len(text) > 240:
            return None
        if message["role"] == "assistant":
            lowered = text.lower()
            if any(phrase in lowered for phrase in REJECT_ASSISTANT_PHRASES):
                return None
            previous_user = messages[index - 1]["content"]
            if previous_user.rstrip(".!?").strip().lower() == text.rstrip(".!?").strip().lower():
                return None
            turn_assistant_tokens = assistant_len(tokenizer, text)
            if turn_assistant_tokens > max_assistant_tokens:
                return None
            assistant_tokens_total += turn_assistant_tokens

    input_tokens = compact_messages_len(tokenizer, messages)
    if input_tokens > max_pair_input_tokens:
        return None

    return {
        "messages": messages,
        "source": "openrouter:nemotron-synth",
        "input_tokens": input_tokens,
        "assistant_tokens": assistant_tokens_total,
        "turns": assistant_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_jsonl", default="data/smoltalk_pair128_train.jsonl")
    parser.add_argument("--extra_seed_jsonl", default=None)
    parser.add_argument("--out", default="data/synth_openrouter_pair128.jsonl")
    parser.add_argument("--tokenizer", default="tokenizer_qwends.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--few_shots", type=int, default=6)
    parser.add_argument("--max_pair_input_tokens", type=int, default=128)
    parser.add_argument("--max_assistant_tokens", type=int, default=48)
    parser.add_argument("--multi_turn_ratio", type=float, default=0.5)
    parser.add_argument("--min_multi_turns", type=int, default=2)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no_response_format", action="store_true")
    parser.add_argument("--exclude_reasoning", action="store_true")
    parser.add_argument("--reasoning_effort", default="none")
    parser.add_argument("--max_attempts", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env_file(".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY was not found in the environment or .env")

    rng = random.Random(args.seed)
    seed_examples = usable_seed_examples(load_jsonl(Path(args.seed_jsonl)))
    if args.extra_seed_jsonl:
        seed_examples.extend(usable_seed_examples(load_optional_jsonl(Path(args.extra_seed_jsonl))))
    if not seed_examples:
        raise RuntimeError("no usable seed examples found")
    tokenizer = load_bpe_tokenizer(args.tokenizer)
    add_chat_special_tokens(tokenizer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[tuple[tuple[str, str], ...]] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        for example in load_jsonl(out_path):
            key = message_key(example)
            if key is not None:
                existing_keys.add(key)

    accepted = 0
    submitted_attempts = 0
    worker_count = max(1, args.threads)
    with out_path.open("a", encoding="utf-8") as file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: set[concurrent.futures.Future[tuple[int, list[Any] | None, str | None]]] = set()

            def submit_request() -> None:
                nonlocal submitted_attempts
                submitted_attempts += 1
                shots = rng.sample(seed_examples, k=min(args.few_shots, len(seed_examples)))
                prompt = build_prompt(
                    shots,
                    min(args.batch_size, args.count - accepted),
                    multi_turn_ratio=args.multi_turn_ratio,
                    min_multi_turns=args.min_multi_turns,
                    max_turns=args.max_turns,
                )
                futures.add(executor.submit(request_values, submitted_attempts, api_key, args, prompt))

            while accepted < args.count:
                while len(futures) < worker_count and accepted < args.count:
                    if args.max_attempts and submitted_attempts >= args.max_attempts:
                        break
                    submit_request()
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                if not futures:
                    raise RuntimeError(
                        f"stopped after {submitted_attempts} attempts with {accepted}/{args.count} accepted examples"
                    )

                done, futures = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    attempt, values, error = future.result()
                    if error is not None:
                        print(f"attempt {attempt}: request/parse failed: {error}", flush=True)
                        continue
                    assert values is not None

                    batch_accepted = 0
                    for value in values[: args.batch_size]:
                        example = validate_example(
                            value,
                            tokenizer,
                            max_pair_input_tokens=args.max_pair_input_tokens,
                            max_assistant_tokens=args.max_assistant_tokens,
                            max_turns=args.max_turns,
                        )
                        if example is None:
                            continue
                        key = message_key(example)
                        assert key is not None
                        if key in existing_keys:
                            continue
                        example["source"] = f"openrouter:{args.model}"
                        existing_keys.add(key)
                        file.write(json.dumps(example, ensure_ascii=False) + "\n")
                        accepted += 1
                        batch_accepted += 1
                        if accepted >= args.count:
                            break

                    file.flush()
                    print(
                        f"attempt {attempt}: accepted {batch_accepted} | total {accepted}/{args.count}",
                        flush=True,
                    )

                    if accepted >= args.count:
                        for pending in futures:
                            pending.cancel()
                        futures.clear()
                        break

                if args.max_attempts and submitted_attempts >= args.max_attempts and not futures and accepted < args.count:
                    raise RuntimeError(
                        f"stopped after {submitted_attempts} attempts with {accepted}/{args.count} accepted examples"
                    )

    print(f"wrote {accepted:,} examples to {args.out}")


if __name__ == "__main__":
    main()
