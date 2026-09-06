#!/usr/bin/env python3
"""Shared FLORES fertility measurement: v0 bugs, corrected metrics, denominators."""

from __future__ import annotations

import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORPUS_DIR = ROOT / "corpus_flores"
LANG_ORDER = ("eng", "hin", "kan", "tam")
LANG_NAMES = {
    "eng": "English",
    "hin": "Hindi",
    "kan": "Kannada",
    "tam": "Tamil",
}


def configure_stdio():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_lines(path: Path, *, nfc: bool = True) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if nfc:
                line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def load_corpora(*, nfc: bool = True) -> dict[str, list[str]]:
    from prepare_flores_corpus import prepare

    prepare()
    return {lang: read_lines(CORPUS_DIR / f"{lang}.txt", nfc=nfc) for lang in LANG_ORDER}


@lru_cache(maxsize=8)
def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        repo_revision = spec[3:]
        if "@" in repo_revision:
            repo, revision = repo_revision.rsplit("@", 1)
            tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        else:
            tok = AutoTokenizer.from_pretrained(repo_revision)
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken

    return tiktoken.get_encoding(spec).encode


def whitespace_words(line: str) -> list[str]:
    return line.split()


def v0_space_words(line: str) -> list[str]:
    return line.split(" ")


def grapheme_clusters(line: str) -> list[str]:
    return regex.findall(r"\X", line)


def utf8_bytes(line: str) -> bytes:
    return line.encode("utf-8")


def measure(lines, encode, *, do_lower: bool, keep_empty: bool, micro: bool) -> dict:
    toks, words, chars, ratios, tpc = [], [], [], [], []
    empty_total = 0
    for line in lines:
        text = line.lower() if do_lower else line
        n_tok = len(encode(text))
        w = v0_space_words(text) if keep_empty else whitespace_words(text)
        n_words = len(w)
        n_chars = len(text)
        empty_total += sum(1 for x in v0_space_words(text) if x == "")
        toks.append(n_tok)
        words.append(n_words)
        chars.append(n_chars)
        ratios.append(n_tok / n_words)
        tpc.append(n_tok / n_chars)
    n = len(lines)
    return {
        "n_lines": n,
        "tokens": sum(toks),
        "words": sum(words),
        "chars": sum(chars),
        "empty_words": empty_total,
        "fertility": (sum(toks) / sum(words)) if micro else (sum(ratios) / n),
        "tok_per_char": (sum(toks) / sum(chars)) if micro else (sum(tpc) / n),
    }


def v0_recipe(lines, encode) -> dict:
    """Exact fertility.py analyze(): lower + split(' ') + macro-average."""
    return measure(lines, encode, do_lower=True, keep_empty=True, micro=False)


def corrected_recipe(lines, encode) -> dict:
    """No lowercasing, no empty words, micro-average."""
    return measure(lines, encode, do_lower=False, keep_empty=False, micro=True)


def denominators(lines, encode) -> dict:
    """Corrected tokenization (no lower) under four denominators."""
    n_tok = n_words = n_graphs = n_bytes = 0
    per_sent = []
    for line in lines:
        nt = len(encode(line))
        n_tok += nt
        n_words += len(whitespace_words(line))
        n_graphs += len(grapheme_clusters(line))
        n_bytes += len(utf8_bytes(line))
        per_sent.append(nt)
    n = len(lines)
    return {
        "tokens": n_tok,
        "sentences": n,
        "tok_per_word": n_tok / n_words,
        "tok_per_grapheme": n_tok / n_graphs,
        "tok_per_utf8_byte": n_tok / n_bytes,
        "tok_per_sentence": n_tok / n,
        "words": n_words,
        "graphemes": n_graphs,
        "utf8_bytes": n_bytes,
        "mean_sentence_tokens": sum(per_sent) / n,
    }


def isolate_bugs(lines, encode) -> dict:
    """Each v0 flag on/off. Fertility = tok/word under that recipe."""

    def fert(**flags):
        return measure(lines, encode, **flags)["fertility"]

    return {
        "v0_all_placed": fert(do_lower=True, keep_empty=True, micro=False),
        "clear_lower_only": fert(do_lower=False, keep_empty=True, micro=False),
        "clear_empty_only": fert(do_lower=True, keep_empty=False, micro=False),
        "clear_macro_only": fert(do_lower=True, keep_empty=True, micro=True),
        "all_cleared": fert(do_lower=False, keep_empty=False, micro=True),
    }


def nfc_probe(path: Path, encode) -> dict:
    """Measure the small, non-zero effect of canonical NFC normalization."""
    raw = read_lines(path, nfc=False)
    nfc = [unicodedata.normalize("NFC", x) for x in raw]
    nfd = [unicodedata.normalize("NFD", x) for x in raw]
    changed_nfc = sum(a != b for a, b in zip(raw, nfc))
    tok_raw = sum(len(encode(x)) for x in raw)
    tok_nfc = sum(len(encode(x)) for x in nfc)
    tok_nfd = sum(len(encode(x)) for x in nfd)
    chars_raw = sum(len(x) for x in raw)
    chars_nfd = sum(len(x) for x in nfd)
    return {
        "lines": len(raw),
        "lines_changed_by_nfc": changed_nfc,
        "tokens_as_shipped": tok_raw,
        "tokens_after_nfc": tok_nfc,
        "tokens_if_nfd": tok_nfd,
        "chars_as_shipped": chars_raw,
        "chars_if_nfd": chars_nfd,
        "token_delta_nfc": tok_nfc - tok_raw,
        "char_delta_nfd": chars_nfd - chars_raw,
    }


def ratio_to_eng(values: dict[str, float]) -> dict[str, float]:
    base = values["eng"]
    return {lang: values[lang] / base for lang in values}
