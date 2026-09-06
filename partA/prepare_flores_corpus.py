#!/usr/bin/env python3
"""Download FLORES-200 dev and cache eng/hin/kan/tam as parallel line files."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_flores"
TARBALL = ROOT / "_flores200_dataset.tar.gz"
URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

LANGS = {
    "eng": "./flores200_dataset/dev/eng_Latn.dev",
    "hin": "./flores200_dataset/dev/hin_Deva.dev",
    "kan": "./flores200_dataset/dev/kan_Knda.dev",
    "tam": "./flores200_dataset/dev/tam_Taml.dev",
}


def prepare() -> Path:
    CORPUS.mkdir(parents=True, exist_ok=True)
    needed = [CORPUS / f"{lang}.txt" for lang in LANGS]
    if all(p.exists() and p.stat().st_size > 0 for p in needed):
        return CORPUS
    if not TARBALL.exists():
        urllib.request.urlretrieve(URL, TARBALL)
    with tarfile.open(TARBALL, "r:gz") as tar:
        for lang, member in LANGS.items():
            text = tar.extractfile(member).read().decode("utf-8")
            (CORPUS / f"{lang}.txt").write_text(text, encoding="utf-8")
        meta = tar.extractfile("./flores200_dataset/metadata_dev.tsv").read().decode("utf-8")
        (CORPUS / "metadata_dev.tsv").write_text(meta, encoding="utf-8")
    return CORPUS


if __name__ == "__main__":
    path = prepare()
    print(f"corpus ready: {path}")
    for lang in LANGS:
        n = sum(1 for line in (path / f"{lang}.txt").read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  {lang}: {n} sentences")
