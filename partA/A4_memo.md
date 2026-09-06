# Part A4 — Tokenizer routing memo

**To:** serving and routing  
**From:** intern, tokenizer audit  
**Evidence:** FLORES-200 `dev`, 997 parallel sentences in English, Hindi, Kannada and Tamil

## Finding

I use **tokens per parallel sentence** as the offline matched-meaning denominator because each aligned row carries approximately the same meaning across languages.


| tokenizer        | English | Hindi          | Kannada         | Tamil           |
| ---------------- | ------- | -------------- | --------------- | --------------- |
| GPT-2            | 25.82   | 192.41 (7.45×) | 350.82 (13.59×) | 398.36 (15.43×) |
| XLM-RoBERTa-base | 29.08   | 36.74 (1.26×)  | 39.72 (1.37×)   | 39.21 (1.35×)   |


I do not use whitespace fertility for routing. For the same FLORES meanings, the averages are 21.02 English words, 15.48 Kannada words and 16.18 Tamil words. Tokens per word therefore changes both the numerator and a language-dependent denominator.

## Recommendation

I recommend rejecting REPORT_v0’s fixed **6× Hindi budget** until the production tokenizer is measured. I also recommend against swapping a tokenizer underneath fixed model weights. The GPT-2 measurements show a large premium with its English-centric vocabulary, while the XLM-R measurements show that this premium is vocabulary-dependent rather than inherent to Indic scripts.

XLM-R does **not** estimate FLM-4B or provide evidence about FLM-4B’s 128k tokenizer. Before changing routing or capacity, I would run the same parallel-sentence analysis with the tokenizer actually served by FLM-4B, then repeat it on production-like chat and code-mixed and Romanized prompts. If the served vocabulary still has a large token premium, I would route to a separately trained Indic-aware model rather than attach a replacement tokenizer to FLM-4B.

## Caveat and guardrail

FLORES is translated Wikimedia prose, not production chat, and neither evaluated tokenizer is FLM-4B’s tokenizer. I treat the results as evidence against v0’s causal claim, not as a production price.

In production, I would monitor **mean input-plus-output tokens per successful task**, split by language/script and matched task class, under the production tokenizer. I would compare Indic/English ratios only within comparable workloads and monitor p95 separately for capacity risk.