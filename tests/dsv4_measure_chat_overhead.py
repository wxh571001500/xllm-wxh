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

"""Measure the fixed xLLM DSV4 user-only chat wrapper token cost."""

import argparse
import json
import runpy
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roundtrip-script", required=True)
    parser.add_argument("--suffix-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-prompt-tokens", required=True, type=int)
    parser.add_argument("--anchor", default="anchor-A")
    parser.add_argument("--repeat-count", default=3600, type=int)
    args = parser.parse_args()

    module = runpy.run_path(args.roundtrip_script)
    suffix = Path(args.suffix_file).read_text(encoding="utf-8")
    prompt = module["build_prompt"](args.anchor, args.repeat_count, suffix)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    raw_no_special = len(tokenizer.encode(prompt, add_special_tokens=False))
    raw_special = len(tokenizer.encode(prompt, add_special_tokens=True))
    print(
        json.dumps(
            {
                "chars": len(prompt),
                "raw_no_special": raw_no_special,
                "raw_special": raw_special,
                "server_prompt_tokens": args.server_prompt_tokens,
                "overhead_from_no_special": args.server_prompt_tokens
                - raw_no_special,
                "overhead_from_special": args.server_prompt_tokens - raw_special,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
