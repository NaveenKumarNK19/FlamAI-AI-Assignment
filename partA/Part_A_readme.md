# Part A — Tokenizer audit

In this folder, I answer A1–A4. I use `fertility_flores.ipynb` as the executable walkthrough and `A4_memo.md` as the ≤1-page decision memo.

## Reproduce

From `submition/partA/`, I run:

```text
python prepare_flores_corpus.py
python fertility_flores_with_bugs.py --tokenizer gpt2
python fertility_flores_without_bugs.py
```

I pinned the dependencies in `../requirements.txt`. The four cached text files let me work with the corpus offline; the first use of XLM-R may download model files.

## A1 — Evaluation corpus

I used the following corpus and configuration:

- Source: FLORES-200 `dev` from Meta NLLB.
- URL: `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`
- Languages: `eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`.
- Size: 997 aligned sentences per language.
- Domains from `metadata_dev.tsv`: Wikinews 348, Wikibooks 301, Wikivoyage 348.
- Preprocessing: UTF-8, blank-line removal, surrounding-whitespace stripping, NFC normalization; no lowercasing or punctuation removal.
- XLM-R revision: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.

Cached-file SHA-256:

```text
eng.txt           65df1e8b38702eafd6fe766413056193cee1759129240798ee7117a94b5732f7
hin.txt           42625af82c355766df7a9ba3d4c58fb164ba7ad9ecbefac7d3534e41d538c29d
kan.txt           79699eccd47c3c8d0596caef5e7db081b7598f3c4cca1c08bc84e9ab2cc9a1bf
tam.txt           272ead3715937613de9b5c305a1b961e434f0d15bb4e4e052977cf99305b3cd6
metadata_dev.tsv  bcf7fe481a1f033be3973e7f1d3869c0e0eb133ebaa9ad8ecad2c3748d78ec38
```

FLORES keeps translated sentence meaning approximately constant, so I use it for an offline matched-meaning comparison. It does not directly represent production chat: it excludes code-mixing, Romanized Indic, OCR/ASR noise and long conversations, while translationese and differences in explicitness can still affect sentence length. It also does not use FLM-4B’s tokenizer. I therefore treat these results as evidence against v0’s generalization, not as an estimate of FLM-4B or a production price.

## A2 — Audit of `fertility.py`

With the commands above, I reproduce the v0 recipe and then clear one choice at a time on the same GPT-2/FLORES inputs.

| language | v0 all placed | clear lower only | clear empty only | clear macro only | all cleared |
|---|---:|---:|---:|---:|---:|
| English | 1.2825 | 1.2367 | 1.2826 | 1.2740 | 1.2285 |
| Hindi | 7.8232 | 7.8225 | 7.8260 | 7.7934 | 7.7957 |
| Kannada | 22.1483 | 22.1467 | 22.9456 | 21.6972 | 22.6683 |
| Tamil | 24.7332 | 24.7314 | 24.8669 | 24.4650 | 24.6165 |

### Flaw 1: lowercasing before tokenization

When I clear only lowercasing, English moves −0.0459 while Indic changes by at most 0.0018. Because GPT-2 is case-sensitive, v0 materially transforms mainly the English input and shrinks the measured Indic/English gap.

### Flaw 2: `split(" ")` creates empty “words”

Under literal-space splitting, I count 1/9/692/101 empty fields in English/Hindi/Kannada/Tamil. Clearing only this bug raises Kannada fertility by +0.7973. The direction shows that empty denominator entries were masking fragmentation.

### Conceptual flaw: macro-average of line ratios

The docstring correctly states that the script averages over lines, but that statistic gives a four-word line and a forty-word line equal weight. When I switch only to `sum(tokens)/sum(words)`, Kannada moves −0.4511 and Tamil moves −0.2682.

### Suspicious but retained: NFC

I retain NFC as deliberate canonical preprocessing. Its measured effect is small but non-zero: relative to the as-shipped text, GPT-2 tokens change by English 0, Hindi +239 (+0.12%), Kannada −51 and Tamil −6. I apply NFC consistently to every headline number. `random.seed(1337)` is unused and changes no result.

The original `tok/char` field is more precisely **tokens per Unicode code point**. In A3, I separately measure user-visible grapheme clusters and UTF-8 bytes.

## A3 — Corrected analysis

In the corrected recipe, I keep original casing, drop empty whitespace fields and micro-average. I report four explicit denominators.

| tokenizer/language | tok/word | tok/grapheme | tok/UTF-8 byte | tok/parallel sentence |
|---|---:|---:|---:|---:|
| GPT-2 English | 1.2285 | 0.2056 | 0.2055 | 25.82 |
| GPT-2 Hindi | 7.7957 | 2.3279 | 0.5946 | 192.41 |
| GPT-2 Kannada | 22.6683 | 4.0588 | 0.9786 | 350.82 |
| GPT-2 Tamil | 24.6165 | 4.2043 | 0.9959 | 398.36 |
| XLM-R English | 1.3837 | 0.2316 | 0.2314 | 29.08 |
| XLM-R Hindi | 1.4888 | 0.4446 | 0.1135 | 36.74 |
| XLM-R Kannada | 2.5666 | 0.4595 | 0.1108 | 39.72 |
| XLM-R Tamil | 2.4227 | 0.4138 | 0.0980 | 39.21 |

For this offline parallel corpus, **tokens per parallel sentence** is my matched-meaning denominator because it holds semantic work approximately constant. It is still a proxy rather than a production measure. For a production decision, I would measure the same quantity with FLM-4B’s actual tokenizer on matched production tasks. XLM-R does not estimate FLM-4B; it only shows that the GPT-2 gap depends on vocabulary.

## A4

In `A4_memo.md`, I recommend blocking the 6× routing assumption until the served tokenizer is measured. I use XLM-R only as evidence that the GPT-2 gap is vocabulary-dependent.

I stored the generated machine-readable values in `results/flores_eval.json`.
