# Part B — Capacity reconciliation

I used `bench/model_spec.md` and `bench/bench_log.csv` to answer B1–B4. To reproduce my results, run:

```text
python kv_capacity.py
python analyze_bench.py
```

`capacity.ipynb` is my portable executable walkthrough.

## B1 — KV cache and maximum concurrency

For GQA, I used the eight KV heads rather than the 24 query heads:

```text
bytes/token = 28 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes
            = 114,688 bytes/token = 112 KiB/token

usable GPU = 24 × 0.92 = 22.08 GB
weights    = 4.2e9 × 2 = 8.40 GB
KV budget  = 22.08 − 8.40 − 1.60 = 12.08 GB
max tokens = 12.08e9 / 114,688 = 105,329
full 4096-token sequences = 105,329 / 4096 = 25.715 → floor 25
```

The decimal-GB interpretation agrees with the log. For long batch 24, I predict 0.933 KV utilization versus 0.93 recorded; for short batch 64, I predict 0.467 versus 0.47. Long batches 32 and 48 would require 1.244× and 1.867× of the pool. In the log, utilization caps at 0.97 and 7 and 23 sequences are preempted.

## B2 — Long-context anomaly

At prompt 3584 and generation 512, I found that mixed throughput peaks at batch 24:


| batch | reported tok/s | generated tok/s | wall s | KV util | sequences preempted |
| ----- | -------------- | --------------- | ------ | ------- | ------------------- |
| 16    | 1311.4         | 163.9           | 49.97  | 0.62    | 0                   |
| 24    | 1607.4         | 200.9           | 61.16  | 0.93    | 0                   |
| 32    | 1384.0         | 173.0           | 94.71  | 0.97    | 7                   |
| 48    | 1298.5         | 162.3           | 151.41 | 0.97    | 23                  |


The anomaly starts when the predicted KV demand exceeds the available pool: cache utilization pins, preemption begins, and wall time rises.

**Change:** I would cap active 4k sequences at 24 with `--max-num-seqs 24` or equivalent admission control. The measured batch-24 row supports an active set near 0.93 KV utilization, no preemption, and approximately 201 generated tok/s. This does **not** show that a 48-request burst has a 69-second p95. With the cap, 24 requests will queue, so I would load-test burst latency before making that claim.

## B3 — Misread throughput and honest goodput

I treated `reported_tok_s` as prompt plus generated tokens:

```text
reported_tok_s = (prompt_len + gen_len) × requests / wall_clock_s
```

For long batch 24, I calculated emitted-token goodput in two equivalent ways:

```text
24 × 512 / 61.16 = 200.92 generated tok/s
1607.4 × 512 / 4096 = 200.93 generated tok/s
```

These are equivalent derivations from the same measured row, not independent experiments. At batch 16, end-to-end generated-token goodput is 294.5 tok/s for the short row and 163.9 tok/s for the long row. Because the rows also use different generation lengths, I used ITL for the cleaner decode comparison: 48.33 ms/token short versus 77.20 ms/token long.

My conclusion is that the harness mixes prefill and decode, capacity peaks before KV overflow, and batch 48 is already measured at 1298.5 mixed / 162.3 generated tok/s, not 3200.

## B4 — Production counter

I would monitor the per-run delta of vLLM’s scheduler preemption counter, commonly exposed as `vllm:num_preemptions_total`, after verifying its name in the deployed vLLM version. I expect zero below the knee and a positive delta once active long-context demand exceeds roughly 25 sequences. The CSV records 7 and 23 **unique sequences preempted at least once** at batches 32 and 48. A cumulative event counter may differ from those values if one sequence is preempted repeatedly. A simultaneous cache-usage gauge near 0.97 would add evidence for the cache-full → preemption mechanism.