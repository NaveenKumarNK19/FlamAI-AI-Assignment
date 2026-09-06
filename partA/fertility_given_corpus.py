#!/usr/bin/env python3
"""
fertility_given_corpus.py -- corrected fertility benchmark on the given corpora

Same job as fertility.py (v0), but with the measurement bugs removed.

Usage (same CLI as v0):
    python fertility_given_corpus.py --corpus eng=corpus_sample/eng_sample.txt \
                                    --corpus hin=corpus_sample/hin_sample.txt \
                                    --tokenizer gpt2

What was removed from v0, and why
---------------------------------
1. line.lower() before encoding
   GPT-2 is case-sensitive. Lowercasing changes English token IDs (NASA,
   Bengaluru, GPU, ...) and is a no-op for Hindi. That silently biases
   the language comparison. Fertility must be measured on the text as served.

2. line.split(" ") for word counts
   split(" ") treats a double space as an empty word. Both given corpora
   contain a planted double space, which inflates the word denominator and
   understates fertility. Standard fertility uses whitespace-delimited
   words: line.split() (any whitespace, empty tokens dropped).

3. Macro-average of per-line ratios
   v0 reports mean(tokens_i / words_i). A 4-word line and a 12-word line
   get equal weight, so short lines dominate. Fertility is defined on the
   corpus: total_tokens / total_words (micro-average). tok/char is the
   same idea: total_tokens / total_chars.

4. unused import random / random.seed(1337) and unused import sys
   Nothing in the pipeline is stochastic. A seed implied a sampling step
   that never existed and made the script look more rigorous than it was.

Kept on purpose
---------------
- NFC normalization: composing Hindi combining marks so the same visual
  word is not counted as two Unicode spellings.
- Whitespace words (not morphological segmentation): this is the standard
  fertility definition, and it matches v0's intent. We only fixed the
  empty-word bug inside that definition.
- GPT-2 as the default tokenizer: this file re-measures the *given*
  corpora fairly. It does not claim GPT-2 is the serving tokenizer.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass, field


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken

    enc = tiktoken.get_encoding(spec)
    return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            lines.append(unicodedata.normalize("NFC", line))
    return lines


def whitespace_words(line: str):
    """Standard fertility denominator: non-empty whitespace-delimited tokens."""
    return line.split()


def v0_space_words(line: str):
    """v0 bug: split only on a literal single space, keeping empty words."""
    return line.split(" ")


@dataclass
class LineStats:
    text: str
    n_tokens: int
    n_words: int
    n_chars: int
    n_empty_space_words: int
    tokens_if_lowercased: int


@dataclass
class CorpusStats:
    lang: str
    path: str
    lines: list[LineStats] = field(default_factory=list)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def tokens(self) -> int:
        return sum(l.n_tokens for l in self.lines)

    @property
    def words(self) -> int:
        return sum(l.n_words for l in self.lines)

    @property
    def chars(self) -> int:
        return sum(l.n_chars for l in self.lines)

    @property
    def fertility(self) -> float:
        return self.tokens / self.words

    @property
    def tok_per_char(self) -> float:
        return self.tokens / self.chars

    @property
    def macro_fertility(self) -> float:
        return sum(l.n_tokens / l.n_words for l in self.lines) / self.n_lines

    @property
    def macro_tok_per_char(self) -> float:
        return sum(l.n_tokens / l.n_chars for l in self.lines) / self.n_lines


def analyze_corpus(lang: str, path: str, encode) -> CorpusStats:
    stats = CorpusStats(lang=lang, path=path)
    for line in read_lines(path):
        words = whitespace_words(line)
        empty = sum(1 for w in v0_space_words(line) if w == "")
        stats.lines.append(
            LineStats(
                text=line,
                n_tokens=len(encode(line)),
                n_words=len(words),
                n_chars=len(line),
                n_empty_space_words=empty,
                tokens_if_lowercased=len(encode(line.lower())),
            )
        )
    return stats


def v0_metrics(lines: list[str], encode):
    """Reproduce fertility.py exactly, for a side-by-side check."""
    fert, tpc = [], []
    total_tok = total_words_buggy = total_chars = 0
    for line in lines:
        lowered = line.lower()
        n_tok = len(encode(lowered))
        n_words = len(v0_space_words(lowered))
        n_chars = len(lowered)
        fert.append(n_tok / n_words)
        tpc.append(n_tok / n_chars)
        total_tok += n_tok
        total_words_buggy += n_words
        total_chars += n_chars
    n = len(fert)
    return {
        "macro_fertility": sum(fert) / n,
        "macro_tpc": sum(tpc) / n,
        "tokens": total_tok,
        "words_including_empty": total_words_buggy,
        "chars": total_chars,
    }


def ablation(lines: list[str], encode):
    """
    Isolate each v0 bug. Cells are fertility (tok/word) under that recipe.

    lower:  encode(line.lower()) vs encode(line)
    empty:  split(" ") vs split()
    avg:    macro (mean of ratios) vs micro (sum/sum)
    """

    def fertility(do_lower: bool, keep_empty: bool, micro: bool) -> float:
        toks, words, ratios = [], [], []
        for line in lines:
            text = line.lower() if do_lower else line
            n_tok = len(encode(text))
            w = v0_space_words(text) if keep_empty else whitespace_words(text)
            n_words = len(w)
            toks.append(n_tok)
            words.append(n_words)
            ratios.append(n_tok / n_words)
        if micro:
            return sum(toks) / sum(words)
        return sum(ratios) / len(ratios)

    return {
        "v0  (lower, empty words, macro)": fertility(True, True, False),
        "fix lower only": fertility(False, True, False),
        "fix empty-words only": fertility(True, False, False),
        "fix averaging only": fertility(True, True, True),
        "fix lower + empty": fertility(False, False, False),
        "fix lower + averaging": fertility(False, True, True),
        "fix empty + averaging": fertility(True, False, True),
        "all fixes (lower off, no empty, micro)": fertility(False, False, True),
    }


def print_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main():
    ap = argparse.ArgumentParser(
        description="Corrected tokenizer fertility on the given sample corpora."
    )
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=corpus_sample/eng_sample.txt",
    )
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    # Windows consoles often default to cp1252; Hindi samples need UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    encode = load_tokenizer(args.tokenizer)
    corpora = []
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        corpora.append(analyze_corpus(lang, path, encode))

    print(f"tokenizer: {args.tokenizer}")
    print("metric:    micro-average on original text (no lowercasing)")
    print("words:     whitespace-delimited, empty tokens dropped")
    print()

    print("=== Corrected corpus totals ===")
    print_table(
        [
            [
                c.lang,
                c.n_lines,
                c.tokens,
                c.words,
                c.chars,
                f"{c.fertility:.4f}",
                f"{c.tok_per_char:.4f}",
            ]
            for c in corpora
        ],
        ["lang", "lines", "tokens", "words", "chars", "fertility (tok/word)", "tok/char"],
    )

    if len(corpora) >= 2:
        base = corpora[0]
        print()
        for other in corpora[1:]:
            fert_ratio = other.fertility / base.fertility
            tpc_ratio = other.tok_per_char / base.tok_per_char
            print(
                f"{other.lang} fertility is {fert_ratio:.2f}x {base.lang} "
                f"({other.fertility:.4f} / {base.fertility:.4f})"
            )
            print(
                f"{other.lang} tok/char  is {tpc_ratio:.2f}x {base.lang} "
                f"({other.tok_per_char:.4f} / {base.tok_per_char:.4f})"
            )
            # If the gap were "Hindi has more characters per word", tok/char
            # would stay similar. A much larger tok/char gap is tokenizer
            # fragmentation, not script length.
            print(
                f"chars/word: {base.lang}={base.chars / base.words:.2f}, "
                f"{other.lang}={other.chars / other.words:.2f} "
                f"(script length does not explain the fertility gap)"
            )

    print()
    print("=== v0 reproduction vs corrected ===")
    cmp_rows = []
    for c in corpora:
        raw_lines = [l.text for l in c.lines]
        v0 = v0_metrics(raw_lines, encode)
        cmp_rows.append(
            [
                c.lang,
                f"{v0['macro_fertility']:.4f}",
                f"{c.fertility:.4f}",
                f"{c.fertility - v0['macro_fertility']:+.4f}",
                f"{v0['macro_tpc']:.4f}",
                f"{c.tok_per_char:.4f}",
                f"{c.tok_per_char - v0['macro_tpc']:+.4f}",
            ]
        )
    print_table(
        cmp_rows,
        [
            "lang",
            "v0 fertility",
            "corrected fertility",
            "delta fert",
            "v0 tok/char",
            "corrected tok/char",
            "delta tpc",
        ],
    )

    print()
    print("=== Ablation: fertility (tok/word) after removing each bug ===")
    for c in corpora:
        raw_lines = [l.text for l in c.lines]
        print(f"\n{c.lang}")
        recipes = ablation(raw_lines, encode)
        print_table(
            [[name, f"{value:.4f}"] for name, value in recipes.items()],
            ["recipe", "fertility"],
        )

    print()
    print("=== Per-line detail (corrected) ===")
    for c in corpora:
        print(f"\n{c.lang}  [{c.path}]")
        print_table(
            [
                [
                    i + 1,
                    l.n_tokens,
                    l.n_words,
                    l.n_chars,
                    f"{l.n_tokens / l.n_words:.3f}",
                    l.n_empty_space_words,
                    l.n_tokens - l.tokens_if_lowercased,
                    l.text[:48] + ("…" if len(l.text) > 48 else ""),
                ]
                for i, l in enumerate(c.lines)
            ],
            [
                "#",
                "tok",
                "words",
                "chars",
                "fert",
                "empty_words_v0",
                "tok_delta_if_lower",
                "text",
            ],
        )
        extra_empty = sum(l.n_empty_space_words for l in c.lines)
        extra_tok_from_case = sum(
            l.n_tokens - l.tokens_if_lowercased for l in c.lines
        )
        print(
            f"  empty words invented by split(' '): {extra_empty}  "
            f"(v0 counted these as real words)"
        )
        print(
            f"  extra tokens on original casing vs lowercased: "
            f"{extra_tok_from_case:+d}  (0 means lowercasing did not change encoding)"
        )


if __name__ == "__main__":
    main()
