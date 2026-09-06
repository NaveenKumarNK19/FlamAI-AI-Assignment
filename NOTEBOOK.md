# Lab notebook

This is my chronological log for the tokenizer, serving, and register take-home. I use the format **hypothesis --> experiment --> result --> revision**. I left the dead ends in because they affected the work I did on 6 Sep 2026.

---

## 6 Sep 2026, morning — understanding the assignment

**Hypothesis.** I expected `G:\VIT\Intern` to be a normal intern repository
with a README and a problem PDF.

**Experiment.** I listed the files recursively, searched for `.md`, `.pdf`,
and `.docx` files, and searched the web for distinctive strings from
`REPORT_v0.md`, including “FLM-4B-Instruct” and “good enough for the deck.”

**Result.** I found six files under `starter_kit/` and no problem-statement
document in the workspace. The web search did not find this take-home. I
therefore treated the starter kit as the brief: it contained a previous
intern's fertility script, two 10-line corpora, a serving CSV, and a
leadership memo that said not to edit the conclusions.

**Revision.** I treated v0 as adversarial and split the work into two tracks:
tokenizer fertility for English and Hindi on GPT-2, and L4 serving analysis
from `bench_log.csv`. I did not trust the conclusions in `REPORT_v0.md`.

---

## Same morning — first pass at `fertility.py`

**Hypothesis.** I expected the reported Hindi fertility of about 7.45 versus
English at about 1.27 to be real, but not the proposed cause that a script
with more characters will challenge any tokenizer. I also suspected planted
issues in `split(" ")`, `.lower()`, and the unused `random.seed`.

**Experiment.** I read `fertility.py` and both sample files, counted double
spaces, and did not run the script yet because `tiktoken` was not installed.

**Result.**

- English line 7 and Hindi line 10 each contained a double space.
- `.lower()` was a no-op on Devanagari but a real transformation on GPT-2
English.
- The docstring said “averaged over lines,” which means a macro-average, not
`total_tokens / total_words`.
- `random` and `sys` were unused.
- NFC initially looked like a silent mutation.

**Revision.** I separated the audit into two code bugs that move the numbers,
one conceptual metric error, and one operation that looked questionable but
was valid: NFC. I did not count the dead seed as a numerical bug.

---

## Late morning — reproducing v0 and correcting the script

**Hypothesis.** I expected the unmodified `fertility.py` to reproduce the
deck's values: English 1.27, Hindi 7.45, and a 5.89× ratio. I expected a
corrected script on the same 10 lines to lower English and raise Hindi,
widening the gap.

**Experiment.** I installed `tiktoken`, ran `fertility.py`, and wrote
`fertility_given_corpus.py` with flags for lowercasing, empty words, and
micro versus macro averaging. I then ablated one flag at a time.

**Result.** The v0 script reproduced the deck exactly. The corrected values
were English **1.2308**, Hindi **7.5246**, and a **6.11×** ratio. The isolated
effects were:

- Lowercasing changed English by **−0.036** and Hindi by **0**. The extra
GPT-2 tokens came from `Quarterly Review`, `NASA`/`ISRO`, and `GPU`.
- Removing empty words changed Hindi line 10 fertility from 7.5 to **9.0**.
- Macro versus micro averaging gave English 1.247 versus 1.231.

**Dead end / surprise.** My first corrected run crashed on Windows cp1252
while printing Hindi. I had to force UTF-8 stdout. This was a display problem,
not a fertility problem, although it initially made the Hindi path look
broken.

**Revision.** The 7.4× tokens-per-character ratio, combined with fewer
characters per word for Hindi (4.75 versus 5.74), contradicted the claim that
Hindi costs more because its words are longer, even on n=10. I still did not
treat n=10 as sufficient for Part A. I wrote `explanation.md` and later
converted the walkthrough to `fertility_given_corpus.ipynb`.

I later revisited whether there were only four issues in the file. My answer
remained yes for `fertility.py`: two code issues, one conceptual averaging
issue, and one dead seed. NFC was not a bug. The serving mistakes were in
`REPORT_v0.md` and the CSV interpretation, not in `analyze()`.

---

## Afternoon — Part A plan and the gated FLORES dead end

**Hypothesis.** I planned to use the 997 parallel sentences in FLORES-200
`dev` for English, Hindi, Kannada, and Tamil, with GPT-2 and
`xlm-roberta-base`. I expected
`datasets.load_dataset("facebook/flores", ...)` to provide the data.

**Experiment.** I installed `datasets`, `transformers`, and `regex`, then
tried `facebook/flores`, `openlanguagedata/flores_plus`, and
`Muennighoff/flores200`.

**Result — dead end.**

- `facebook/flores` was **gated** and required a Hugging Face token.
- `openlanguagedata/flores_plus` was also **gated**.
- `Muennighoff/flores200` used loading scripts that are **no longer
supported**; the hub contained only `flores200.py` and a README.
- `tinyurl.com/flores200dataset` returned an HTML preview rather than a
tarball.

**Revision.** I stopped depending on a Hugging Face login and used Meta's
public tarball:
`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`. Its HEAD
request returned 200 and its size was 25 MB. I extracted only
`dev/{eng_Latn,hin_Deva,kan_Knda,tam_Taml}.dev`.

**Second extraction dead end.** My first extraction loop printed Hindi and
then died on cp1252, so Kannada and Tamil were not written. I reran it with
`PYTHONIOENCODING=utf-8` and obtained 997 lines per language. The metadata
domains were wikinews 348, wikibooks 301, and wikivoyage 348. The FLORES
papers say WikiJunior, but the tarball says wikibooks, so I recorded what was
actually on disk.

---

## Evening — Part A numbers and surprises

**Hypothesis A.** I expected the empty-word bug to be small on FLORES,
perhaps a few double spaces like those in the planted sample.

**Hypothesis B.** I expected the GPT-2 token delta between NFC and the
as-shipped text to be approximately zero, which would have supported calling
NFC harmless.

**Hypothesis C.** I expected GPT-2 Hindi tokens per word to be about 6–8×
English. I expected a multilingual tokenizer to improve that ratio, perhaps
to around 3×.

**Experiment.** I ran `fertility_flores_with_bugs.py` and
`fertility_flores_without_bugs.py` on the cached 997-sentence corpus. I
isolated flags and probed NFC, including an NFD comparison. I used word,
grapheme (`regex \X`), UTF-8 byte, and parallel sentence denominators with
GPT-2 and `xlm-roberta-base`.

**Result — surprise 1, empty words.** Kannada `split(" ")` created **692**
empty strings, compared with Tamil 101, Hindi 9, and English 1. Disabling
that behavior moved Kannada fertility by **+0.80**. The 10-line sample had
substantially understated this bug because FLORES Kannada contains many
double spaces.

**Result — surprise 2, NFC.** NFC changed **90 Hindi lines** and added
**239** GPT-2 tokens, about **0.12%**. NFD would add **4,818 characters** and
about 14k tokens to Kannada. I kept NFC as the appropriate canonical
preprocessing, but changed the rationale from “a no-op” to “small relative
to NFD.” Calling NFC itself a distorting bug would have been incorrect.

**Result — surprise 3, denominator.** Whitespace words per sentence were
English 21.0, Hindi 24.7, Kannada **15.5**, and Tamil **16.2** for the same
997 meanings. Tokens per word therefore overstates Dravidian cost. GPT-2
tokens per parallel sentence, relative to English, were Hindi **7.45×**,
Kannada **13.6×**, and Tamil **15.4×**.

**Result — surprise 4, routing implication.** XLM-RoBERTa tokens per sentence,
relative to English, were Hindi **1.26×**, Kannada **1.37×**, and Tamil
**1.35×**. This showed that the 6× Indic budget was a GPT-2 vocabulary
result, not a universal property of the script. The analysis code itself did
not change.

**Test I did not run.** I did not train or download FLM-4B's 128k tokenizer.
That would have been the vocabulary used in serving, but it was outside the
constraints. I kept this as a caveat rather than substituting an estimated
number.

**Revision.** I used **tokens per sentence under the tokenizer being served**
as the A3 headline, wrote the A4 memo around that finding, and added
`Part_A_readme.md`.

---

## Evening — Part B and the GiB-versus-GB dead end

**Hypothesis.** I calculated KV bytes per token as
`28 × 8 × 128 × 2 × 2 = 114,688`. Treating 24 GB as 24 GiB suggested a
capacity of about 28 sequences at length 4096. I expected the long batch-24
row at `kv_cache_util=0.93` to reveal which unit the specification used.

**Experiment.** I calculated capacity in both unit systems and compared
predicted utilization,
`n × seq_len × 114688 / KV_budget`, with every CSV row.

**Result.** Decimal GB gave a KV budget of
`22.08 − 8.4 − 1.6 = 12.08 GB`, a maximum of 105,329 tokens, and **25.7
sequences**. It matched the log: the long batch-24 row predicted **0.933**
versus the observed 0.93, while the short batch-64 row predicted **0.467**
versus 0.47. GiB predicted about 0.86 at batch 24 and did not match. Long
batch 32 and 48 predicted occupancy of 1.24 and 1.87; the log capped
utilization at 0.97 and reported 7 and 23 preempted sequences.

**Wrong turn I avoided.** Using 24 query heads would give `344,064`
bytes per token, predict only about eight concurrent 4k sequences, and fail
to explain the clean batch-24 run.

**B2 result.** Long `reported_tok_s` peaked at 1607 for batch 24 and then
fell to 1298 at batch 48. The v0 value of about 3200 came from the naive
extrapolation `1607 × 2 ≈ 3215`. The observed mechanism was a full KV cache
followed by preemption.

**B3 hypothesis.** I expected
`reported_tok_s = (prompt+gen)*n/wall`.

**Experiment.** I reconstructed every row.

**Result.** The largest difference was no more than 0.2 tok/s, consistent
with rounding. For the long batch-24 row, generation goodput was:

- `24 × 512 / 61.16 = 200.92`
- `1607.4 × 512 / 4096 = 200.93`

At batch 16, generation throughput was 294 tok/s for short requests and
164 tok/s for long requests, the opposite of the v0 conclusion.

**Surprise / almost-wrong interpretation.** ITL implied a decode rate of
**250 tok/s**, not 201 tok/s, because wall-clock time includes prefill. The
two calculations for generation goodput are equivalent derivations from
different logged columns, not independent experiments. I retained ITL only
as a separate decode-rate view.

**Revision.** I recommended `--max-num-seqs 24` for 4k traffic. I did not
make fp8 KV the primary B2 fix because the log did not establish the required
runtime support. I selected `vllm:num_preemptions_total` as the live counter.

---

## Late evening — Part C and the available A100

**Initial hypothesis.** I nearly chose path (a), synthetic casual SFT on the
4B model, because the plan had two weeks of A100 access and six target
languages.

**Why I rejected it.** The reviewer covered only Hindi and Kannada for
10 hours per week. A launch in three weeks, with week 3 reserved for freeze,
left about **20 labeling hours**. At two minutes per item, that is **600
labels**. My first pass left about 200 train pairs per reviewable language
after holdout; the later audit reduced that to about 170 after accounting
for the day-1 pilot. It left **zero gold labels** for Tamil, Telugu, Bengali,
and Marathi. With no API
budget, the training data would be self-rewrites. I judged SFT on unreviewed
self-rewrites too likely to produce textbook language with superficial
slang. The available compute was 336 GPU-hours while LoRA required about
6 hours, so GPU capacity was not the limiting factor.

**Second hypothesis.** I considered path (b), keeping the 4B model frozen and
using a 1B rewriter.

**Why I rejected it using Part B.** The L4 batch-1 ITL was 43 ms. A
150-token rewrite would take about **6.5 seconds** at 4B speed. My initial
1B estimate was **2.3 seconds**, but I had not benchmarked that model, so I
later kept it as a planning assumption rather than evidence. A second decode
would still impose a permanent latency cost, while paraphrasing could also
introduce factual drift.

**Revision.** I recommended an instrumented version of **(c),
prompt-engineering**:

- Success means mean casualness ≥ 3.5/5, at least 70% of items scoring ≥3,
and meaning-ok ≥95% on 100 Hindi plus 100 Kannada items.
- I would stop pure (c) on **day 7** if the mean improvement were below 0.8
or meaning-ok were below 95%, then use Hindi-and-Kannada-only LoRA.
- On day 1, I planned 30 Hindi plus 30 Kannada queries and one two-hour
reviewer session.

I also rejected the vague version of (c), “just add a system prompt,” because
it had no measurable threshold or decision date.

---

## Same night — assembling the submission

**Hypothesis.** I expected the requested `your-submission/` package to
contain `NOTEBOOK.md`, `partA/`, `partB/`, `partC/memo.md`, and
`AI_USAGE.md`.

**Experiment.** I assembled the package in the requested `submition/` folder.
I copied the runnable code and cached FLORES `*.txt` files, but not the 25 MB
tarball, `__pycache__`, or gated Hugging Face downloads.

**Result.** I recorded the layout in this folder's `README.md`. I retained
the dead ends involving gated FLORES, encoding crashes, GiB versus GB, NFC
not being exactly zero, ITL not equaling wall-clock goodput, and the tempting
but unsupported A100-SFT route.

---

## What I still do not know

- I do not know FLM-4B's actual 128k-tokenizer fertility on FLORES.
- I do not know whether vLLM's block size explains `kv_cache_util` reaching
0.97 rather than 1.00; I treated 0.97 as full.
- I did not test whether a casual system prompt changes FLM-4B because I did
not have the model or the reviewer. The Part C day-1 experiment was
designed to answer that question.
- I had no speaker coverage for Telugu, Bengali, or Marathi casual quality.

---

## Submission audit — packaging and claim corrections

**Hypothesis.** I expected the final folder to contain every artifact named
in the README and the Part B and Part C predictions to follow directly from
the measured rows.

**Experiment.** I compared the README paths with the final tree, ran all
Python entry points, checked notebook portability, and recomputed the Part C
reviewer schedule.

**Result.**

- Five files named in the README had not been exported: the Part A and Part B
notebooks and write-ups, including the required A4 memo.
- The notebooks contained machine-specific fallback paths.
- `60 queries × 2 comparisons × 2 min = 4 h`, not the stated 2 h.
- A 24-sequence cap predicts active-set behavior from batch 24, but it does
not predict p95 latency for 48 submitted requests because half will queue.
- NFC changed Hindi by +239 GPT-2 tokens, so “approximately zero” was too
broad.
- XLM-R demonstrated vocabulary dependence but did not estimate FLM-4B's
tokenizer.

**Revision.** I added portable notebooks and written answers under `partA/`
and `partB/`, pinned the dependencies, separated the Part C pilot from the
untouched holdout, reduced day 1 to one comparison per query, and labeled
unmeasured capacity and model claims as extrapolations. I left the unresolved
FLM-4B and four-language reviewer gaps explicit rather than filling them with
proxy claims.