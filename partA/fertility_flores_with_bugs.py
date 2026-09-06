#!/usr/bin/env python3
"""
fertility_flores_with_bugs.py

v0 recipe from fertility.py, applied to the FLORES-200 eval corpus.
Bugs are left in on purpose so they can be compared with the corrected script.

    python fertility_flores_with_bugs.py
    python fertility_flores_with_bugs.py --tokenizer gpt2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flores_metrics import (
    CORPUS_DIR,
    LANG_ORDER,
    configure_stdio,
    load_tokenizer,
    read_lines,
    v0_recipe,
)
from prepare_flores_corpus import prepare


def main():
    configure_stdio()
    prepare()
    ap = argparse.ArgumentParser(description="v0 (buggy) fertility on FLORES-200")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()
    encode = load_tokenizer(args.tokenizer)

    print("recipe:     v0 (lower + split(' ') + macro-average of line ratios)")
    print(f"tokenizer:  {args.tokenizer}")
    print("corpus:     FLORES-200 dev (997 parallel sentences)")
    print()
    print(f"{'lang':<8}{'fertility (tok/word)':>22}{'tok/char':>12}")
    print("-" * 42)

    results = {}
    for lang in LANG_ORDER:
        path = CORPUS_DIR / f"{lang}.txt"
        lines = read_lines(path, nfc=True)
        m = v0_recipe(lines, encode)
        results[lang] = m
        print(f"{lang:<8}{m['fertility']:>22.2f}{m['tok_per_char']:>12.3f}")

    print()
    base = results["eng"]["fertility"]
    for lang in LANG_ORDER[1:]:
        ratio = results[lang]["fertility"] / base
        print(
            f"{lang} is {ratio:.2f}x the fertility of eng "
            f"({'worse' if ratio > 1 else 'better'} tokenization)"
        )


if __name__ == "__main__":
    main()
