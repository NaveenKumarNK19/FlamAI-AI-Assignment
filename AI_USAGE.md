# AI usage

I completed this take-home in **Cursor** with a Grok 4.6 agent on
6 Sep 2026. I chose the EN/HI/KN/TA language set, approved the Part A and
Part B plans before implementation, and made the final Part C recommendation.
The agent wrote and ran most of the code, notebooks, and initial prose. The
reported numbers came from scripts and checked outputs rather than being
invented in the write-up.

## Where AI helped

The agent:

- explored `starter_kit/` and audited `fertility.py`;
- implemented the corrected fertility pipeline, the FLORES analysis, the
  serving arithmetic, and the Part C resource envelope;
- downloaded FLORES-200 from Meta's public tarball after the Hugging Face
  datasets route failed because of gating and deprecated loading scripts;
- generated the `.ipynb` walkthroughs and initial README drafts; and
- ran the local Python work with `tiktoken`, `transformers`, and `regex`.

I used the results to frame the main conclusions: the v0 deck was not
reliable, NFC normalization was not itself a bug, a six-language SFT plan was
not supportable with the available review coverage, and a 1B rewriter would
add an unacceptable serving cost.

## Where AI misled me

AI output also introduced or repeated four claims that did not survive
checking:

- **Part C two-hour arithmetic.** The draft treated
  `60 queries × 2 comparisons × 2 min` as 2 hours. It is 4 hours. I corrected
  the plan by reducing day 1 to one comparison per query and separating the
  pilot from the untouched holdout.
- **Batch-48 latency extrapolation.** A 24-sequence cap explains active-set
  behavior from the measured batch-24 row, but it does not establish p95
  latency for 48 submitted requests. Half of those requests would queue, and
  that queued latency was not measured.
- **NFC “approximately zero” wording.** NFC changed 90 Hindi lines and added
  239 GPT-2 tokens, about 0.12%. That is small relative to the NFD effect, but
  it is not zero. I changed the wording to state the measured result.
- **XLM-R as an FLM proxy.** XLM-R showed that fertility depends strongly on
  tokenizer vocabulary, but it does not estimate FLM-4B's tokenizer. I kept
  the XLM-R result and removed the unsupported implication about FLM-4B.

## How I verified corrections

A later Cursor pass with GPT-5.6 Sol compared the completed folder with the
assignment, executed every Python entry point, checked notebook paths, and
recomputed the Part C schedule. I then kept claims only where the scripts,
CSV rows, or checked notebook outputs supported them. In particular, I
reconstructed `reported_tok_s` from every serving row, compared both GB and
GiB capacity calculations with observed `kv_cache_util`, and checked the NFC
token delta directly on the cached FLORES files.

The same audit found that five files named in the README had not been
exported: the Part A and Part B notebooks and write-ups, including the A4
memo. The agent created portable versions from the existing scripts and
results, removed machine-specific fallback paths, pinned dependencies, and
regenerated the checked-in outputs.

## What I did not test

I did not run FLM-4B or obtain its actual 128k tokenizer, so I did not measure
FLM-4B fertility or verify that a casual system prompt changes its output. I
did not run the proposed native-speaker review, and I had no speaker coverage
for Telugu, Bengali, Marathi, or Tamil. I also did not measure batch-48 p95
latency under a 24-sequence cap or test an fp8 KV-cache configuration.

No public OpenAI, Anthropic, or similar LLM API was used to create casual
parallel data. Part C had no external API budget. FLORES and XLM-R came from
public Meta and Hugging Face artifacts; no FLM-4B or native-speaker result was
fabricated.

## Tools and packages

I used Cursor agents, Python 3.10, `tiktoken`, `transformers`, `regex`,
`nbformat`, and `nbclient`. I installed `datasets` but abandoned that route
when `facebook/flores` and `flores_plus` proved gated.
