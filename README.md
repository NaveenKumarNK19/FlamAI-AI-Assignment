# Submission

I organized this take-home package to match the brief's
`your-submission/` layout.

```text
FlamAI-AI-Assignment/
  NOTEBOOK.md          chronological lab log
  AI_USAGE.md          how I used Cursor and its agent
  README.md            this file
  requirements.txt     pinned runtime dependencies
  partA/               tokenizer audit, notebook, corpus, and A4 memo
  partB/               capacity calculations, notebook, and written answers
  partC/memo.md        ≤1-page register decision
  starter_kit/         given files
```

## Deliverables


| Deliverable  | Path                                                                  |
| ------------ | --------------------------------------------------------------------- |
| Lab notebook | `NOTEBOOK.md`                                                         |
| Part A       | `partA/` (`fertility_flores.ipynb`, `A4_memo.md`, `Part_A_readme.md`) |
| Part B       | `partB/` (`capacity.ipynb`, `Part_B_readme.md`)                       |
| Part C       | `partC/memo.md`                                                       |




## How to re-run

```text
python -m pip install -r requirements.txt

# Part A (needs tiktoken, transformers, regex; FLORES txt already cached)
cd partA
python fertility_flores_with_bugs.py
python fertility_flores_without_bugs.py

# Part B (stdlib only)
cd ../partB
python kv_capacity.py
python analyze_bench.py

# Part C envelope
cd ../partC
python resource_envelope.py
```

I kept the FLORES language files in `partA/corpus_flores/`. I did not include
the 25 MB tarball in the zip; `prepare_flores_corpus.py` can fetch it again if
a file is missing. I first tried Hugging Face `facebook/flores`, but it is
gated, so that route was a dead end (see `NOTEBOOK.md`).

The notebooks resolve files relative to `partA/` and `partB/` and contain no
machine-specific absolute paths. I regenerated the checked-in JSON and
notebook outputs with the commands above.
