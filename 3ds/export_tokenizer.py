#!/usr/bin/env python3
"""Convert a HuggingFace tokenizers JSON into the .cbin format the 3DS runtime reads."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from train import export_tokenizer_cbin

    export_tokenizer_cbin(args.tokenizer_json, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
