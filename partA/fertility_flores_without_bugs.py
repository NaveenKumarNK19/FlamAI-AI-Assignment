#!/usr/bin/env python3
"""
fertility_flores_without_bugs.py

Corrected fertility on FLORES-200: English, Hindi, Kannada, Tamil.

Clears the v0 code bugs, isolates each one's effect, measures NFC's effect,
then reports two tokenizers x four denominators.

    python fertility_flores_without_bugs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from flores_metrics import (
    CORPUS_DIR,
    LANG_NAMES,
    LANG_ORDER,
    configure_stdio,
    corrected_recipe,
    denominators,
    isolate_bugs,
    load_corpora,
    load_tokenizer,
    nfc_probe,
    ratio_to_eng,
    v0_recipe,
)
from prepare_flores_corpus import prepare

TOKENIZERS = {
    "gpt2": "gpt2",
    "xlm-roberta-base": (
        "hf:xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    ),
}
RESULTS = Path(__file__).resolve().parent / "results" / "flores_eval.json"


def table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main():
    configure_stdio()
    prepare()
    corpora = load_corpora(nfc=True)
    payload = {"tokenizers": {}, "nfc": {}, "bugs_gpt2": {}}

    print("=== A2  Isolate v0 bugs on FLORES-200 (tokenizer = gpt2) ===")
    encode_gpt2 = load_tokenizer("gpt2")
    bug_rows = []
    for lang in LANG_ORDER:
        iso = isolate_bugs(corpora[lang], encode_gpt2)
        payload["bugs_gpt2"][lang] = iso
        bug_rows.append(
            [
                lang,
                f"{iso['v0_all_placed']:.4f}",
                f"{iso['clear_lower_only']:.4f}",
                f"{iso['clear_empty_only']:.4f}",
                f"{iso['clear_macro_only']:.4f}",
                f"{iso['all_cleared']:.4f}",
            ]
        )
    table(
        [
            "lang",
            "v0 placed",
            "clear lower",
            "clear empty words",
            "clear macro",
            "all cleared",
        ],
        bug_rows,
    )
    print()
    print("Direction on this corpus (gpt2 tok/word):")
    for lang in LANG_ORDER:
        iso = payload["bugs_gpt2"][lang]
        d_lower = iso["clear_lower_only"] - iso["v0_all_placed"]
        d_empty = iso["clear_empty_only"] - iso["v0_all_placed"]
        d_macro = iso["clear_macro_only"] - iso["v0_all_placed"]
        d_all = iso["all_cleared"] - iso["v0_all_placed"]
        print(
            f"  {lang}: lower {d_lower:+.4f}  empty {d_empty:+.4f}  "
            f"macro {d_macro:+.4f}  all {d_all:+.4f}"
        )

    print()
    print("=== A2  NFC probe (deliberate canonical preprocessing) ===")
    nfc_rows = []
    for lang in LANG_ORDER:
        probe = nfc_probe(CORPUS_DIR / f"{lang}.txt", encode_gpt2)
        payload["nfc"][lang] = probe
        nfc_rows.append(
            [
                lang,
                probe["lines_changed_by_nfc"],
                probe["token_delta_nfc"],
                probe["char_delta_nfd"],
                probe["tokens_as_shipped"],
                probe["tokens_if_nfd"],
            ]
        )
    table(
        [
            "lang",
            "lines NFC changes",
            "token delta NFC",
            "extra chars if NFD",
            "tok shipped",
            "tok if NFD",
        ],
        nfc_rows,
    )
    hin_nfc = payload["nfc"]["hin"]
    hin_pct = 100 * hin_nfc["token_delta_nfc"] / hin_nfc["tokens_as_shipped"]
    print(
        "NFC is deliberate canonical preprocessing, not a no-op: "
        f"Hindi changes by {hin_nfc['token_delta_nfc']:+d} GPT-2 tokens "
        f"({hin_pct:+.2f}%). The effect is small but non-zero; "
        "all headline metrics consistently use NFC text."
    )

    print()
    print("=== A3  Two tokenizers x four denominators (bugs cleared) ===")
    for tok_name, spec in TOKENIZERS.items():
        encode = load_tokenizer(spec)
        payload["tokenizers"][tok_name] = {}
        print(f"\n--- tokenizer: {tok_name} ---")
        rows = []
        sent = {}
        for lang in LANG_ORDER:
            d = denominators(corpora[lang], encode)
            payload["tokenizers"][tok_name][lang] = d
            sent[lang] = d["tok_per_sentence"]
            rows.append(
                [
                    f"{lang} ({LANG_NAMES[lang]})",
                    d["tokens"],
                    f"{d['tok_per_word']:.4f}",
                    f"{d['tok_per_grapheme']:.4f}",
                    f"{d['tok_per_utf8_byte']:.4f}",
                    f"{d['tok_per_sentence']:.2f}",
                ]
            )
        table(
            [
                "lang",
                "tokens",
                "tok/word",
                "tok/grapheme",
                "tok/utf8-byte",
                "tok/sentence",
            ],
            rows,
        )
        rel = ratio_to_eng(sent)
        print("tokens/sentence relative to English:")
        for lang in LANG_ORDER:
            print(f"  {lang}: {rel[lang]:.2f}x  ({sent[lang]:.2f} tok/sent)")

        v0_vs = []
        for lang in LANG_ORDER:
            v0 = v0_recipe(corpora[lang], encode)
            corr = corrected_recipe(corpora[lang], encode)
            v0_vs.append(
                [
                    lang,
                    f"{v0['fertility']:.4f}",
                    f"{corr['fertility']:.4f}",
                    f"{corr['fertility'] - v0['fertility']:+.4f}",
                    f"{payload['tokenizers'][tok_name][lang]['tok_per_sentence']:.2f}",
                ]
            )
        print()
        table(
            ["lang", "v0 tok/word", "corrected tok/word", "delta", "tok/sentence"],
            v0_vs,
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
