#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build and optionally send exact-token DSV4 Store boundary prompts.

The prompt length is measured with the model tokenizer and the same chat
template options used by the xLLM OpenAI endpoint.  Each target gets a unique
marker before the padding so different boundary cases cannot prefix-hit one
another accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


TARGETS = (2047, 2048, 2049, 4095, 4096, 4097)
EXPECTED = "BOUNDARY_OK"
CHAT_TEMPLATE_OVERHEAD = 4


def chat_token_count(tokenizer: Any, content: str) -> int:
    # DeepSeek-V4 uses xLLM's C++ chat template; the checkpoint tokenizer does
    # not carry a Hugging Face chat_template. A user-only request with
    # enable_thinking=False adds a fixed wrapper around the raw content.
    return (
        len(tokenizer.encode(content, add_special_tokens=False))
        + CHAT_TEMPLATE_OVERHEAD
    )


def build_exact_prompt(tokenizer: Any, target: int, marker: str) -> str:
    prefix = (
        f"DSV4 Store boundary marker: {marker}.\n"
        "The following repeated words are inert cache padding. "
        "Ignore them and follow only the final Chinese instruction.\n"
    )
    suffix = (
        "\nPadding ends here. Ignore the padding above. "
        "Reply with exactly: BOUNDARY_OK"
    )
    base = chat_token_count(tokenizer, prefix + suffix)
    if base >= target:
        raise ValueError(f"target {target} is not larger than base {base}")

    units = (" cache", " token", " test", " data", " x", " 0")
    tails = ("", " a", " b", " x", " 0", "!", "。", "\n")
    for unit in units:
        def render(repeats: int, tail: str = "") -> str:
            return prefix + unit * repeats + tail + suffix

        low = 0
        high = max(1, target - base + 64)
        while chat_token_count(tokenizer, render(high)) < target:
            high *= 2
        while low < high:
            middle = (low + high) // 2
            if chat_token_count(tokenizer, render(middle)) < target:
                low = middle + 1
            else:
                high = middle
        for repeats in range(max(0, low - 96), low + 97):
            for tail in tails:
                prompt = render(repeats, tail)
                actual = chat_token_count(tokenizer, prompt)
                if actual == target:
                    return prompt
    raise RuntimeError(f"could not construct an exact {target}-token prompt")


def send_prompt(endpoint: str, model: str, prompt: str, request_id: str) -> dict:
    response = requests.post(
        f"{endpoint}/v1/chat/completions",
        headers={"X-Request-ID": request_id},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    result = {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "request_id": request_id,
    }
    normalized = content.strip().rstrip(".!。！")
    if result["finish_reason"] != "stop" or normalized != EXPECTED:
        raise RuntimeError(f"boundary response failed accuracy: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default=",".join(map(str, TARGETS)))
    parser.add_argument("--marker-prefix", default="dsv4-store-boundary")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--phase", choices=("build", "cold", "replay"), default="build")
    args = parser.parse_args()

    targets = tuple(int(value) for value in args.targets.split(",") if value)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )

    summary = []
    for target in targets:
        marker = f"{args.marker_prefix}-{target}"
        prompt = build_exact_prompt(tokenizer, target, marker)
        prompt_path = output_dir / f"prompt_{target}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        item = {
            "target": target,
            "offline_prompt_tokens": chat_token_count(tokenizer, prompt),
            "prompt_file": str(prompt_path),
        }
        if args.phase != "build":
            if not args.endpoint:
                parser.error("--endpoint is required for cold/replay")
            request_id = f"{args.marker_prefix}-{args.phase}-{target}"
            response = send_prompt(args.endpoint, args.model, prompt, request_id)
            if response["prompt_tokens"] != target:
                raise RuntimeError(
                    f"server/tokenizer token mismatch for {target}: {response}"
                )
            item["response"] = response
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))

    (output_dir / f"{args.phase}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
