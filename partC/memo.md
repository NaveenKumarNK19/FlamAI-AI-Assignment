# Part C — Decision memo

**To:** product / applied  
**From:** intern  
**Re:** casual register in HI, KN, TA, TE, BN, MR.  
**Decision:** I recommend **(c) prompt-engineering first**. I would not SFT the 4B on six-language synthetic data or ship a ≤1B rewriter on the L4. If (c) misses the week-1 kill criterion, my fallback is a **HI+KN-only LoRA** (narrow a), without unreviewed TA/TE/BN/MR data.

## Assumptions

1. I assume formality is mainly **register** (pronouns, verb endings, Sanskritized vs colloquial lexicon), not a missing capability. FLM-4B likely knows casual HI/KN but defaults to textbook language because its pretraining is news/wiki.
2. I treat the reviewer as the only gold source for casualness. They cover **Hindi and Kannada only**, so TA/TE/BN/MR remain unverified.
3. With no API budget, any “synthetic casual” data must be a **self-rewrite from FLM-4B** and may repeat its existing style errors.
4. I assume production remains on the **Part B L4**; a second decode adds latency and uses capacity.
5. I treat the launch review in 3 weeks as a go/no-go. Week 3 is for freezing the candidate, not more labeling.
6. I require casual rewrites to preserve facts and meaning.

## Back-of-envelope

**Reviewer.** I have 10 h/week × 2 labeling weeks (week 3 = launch) = **20 h**. At 2 min/judgment (one pair + 1–5 casual + meaning-ok), that is **30 judgments/h → 600 total**. I reserve **200** for an untouched final holdout (100/lang) and **60** for a separate day-1 pilot (**60 judgments = 2 h**). This leaves at most **340** (~170/lang) for prompt iteration or reviewer-accepted LoRA pairs. **TA/TE/BN/MR gold = 0.**

**GPU.** I have 2 weeks A100-80GB ≤ **336 GPU-h**. My unmeasured planning estimates, not benchmarks, are ~6 GPU-h for 10k self-rewrites, ~6 h for a 4B LoRA, and ~8 h for a 1B rewriter. Reviewer coverage is the tighter constraint.

**Serving (b).** Part B shows 4B ITL from **43 ms/token unloaded to ~96 ms/token near the long-context knee**. A 150-token second decode would therefore cost 6.5–14.4 s at 4B speed. This is an estimate, not a measured rewriter result. A 1B rewriter may be faster, but its latency is unmeasured; I would benchmark it before using latency as a decision fact. It also adds TTFT/KV and paraphrase risk.

**(a) full SFT.** I cannot use 600 reviewed pairs to supervise six languages. SFT on unreviewed self-rewrites would leave TA/TE/BN/MR style quality unverified. Fine-tuning the served 4B may also regress English/code behavior that I cannot re-benchmark well in 3 weeks.

## Success metric (numeric)

On the untouched **100 HI + 100 KN** holdout, I will collect native 1–5 casualness scores (1 = textbook, 5 = natural spoken) plus binary meaning-ok:

- mean casualness **≥ 3.5**, and **≥ 70%** of items **≥ 3**
- meaning-ok **≥ 95%**
- no worse than baseline on a tiny English/code smoke set (0 regressions I would notice in 30 prompts)

I will take (c) to launch review only if both HI and KN pass. I will not claim six-language success: TA/TE/BN/MR remain unverified and need native validation.

## Kill criterion

I will **abandon pure (c) at the end of week 1 (day 7)** if the casual prompt’s mean casualness is **< +0.8** vs baseline, **or** meaning-ok is **< 95%**. I will then use week 2 for a **HI+KN LoRA** trained only on reviewer-accepted self-rewrites (up to ~170/lang after pilot and holdout) and evaluate it once on the untouched holdout on day 12. I will abandon the LoRA if meaning-ok < 90% or the English smoke test breaks. I will not start (b) without a measured L4 latency test and product approval.

## Day-1 experiment

I will use a separate **30 HI + 30 KN pilot set**, never the final holdout. I will generate a baseline and one casual system-prompt candidate (explicit pronoun policy; ban textbook openers). One pairwise judgment per query at 2 min each gives **60 judgments = 2 h**. On day 2, I will iterate or add a few-shot candidate using validation data. After prompt selection, I will evaluate exactly once on the untouched 100+100 holdout.