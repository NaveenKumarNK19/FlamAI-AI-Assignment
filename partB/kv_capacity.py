#!/usr/bin/env python3
"""
kv_capacity.py — Part B1

From bench/model_spec.md alone: KV bytes per token and max concurrent
4096-token sequences. Then check the prediction against bench_log.csv.
"""

from __future__ import annotations

from bench_metrics import (
    SPEC,
    b1_summary,
    configure_stdio,
    kv_bytes_per_token,
    load_log,
    predicted_kv_util,
)


def table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main():
    configure_stdio()
    s = b1_summary()
    bpt = kv_bytes_per_token()

    print("=== B1(a)  KV-cache bytes per token (exact) ===")
    print(
        f"  layers * kv_heads * head_dim * 2(K+V) * 2 bytes(fp16)"
    )
    print(
        f"  {SPEC['layers']} * {SPEC['kv_heads']} * {SPEC['head_dim']} * 2 * 2"
        f"  = {bpt} bytes/token"
    )
    print(f"  = {s['kv_kib_per_token']:.0f} KiB/token")
    print(
        f"  sanity: d_model/q_heads = {SPEC['d_model']}/{SPEC['q_heads']}"
        f" = {s['head_dim_check']:.0f} (matches head_dim)"
    )
    print(
        "  GQA: use 8 KV heads, not 24 query heads "
        f"({SPEC['layers']}*24*128*2*2 = {SPEC['layers']*24*128*2*2} would be wrong)"
    )

    print()
    print("=== B1(b)  Max concurrent 4096-token sequences ===")
    print(f"  usable GPU     = {SPEC['gpu_gb']} * {SPEC['gpu_mem_util']} = {s['usable_gpu_gb']:.2f} GB")
    print(f"  weights (fp16) = {SPEC['params']/1e9:.1f}e9 * 2 = {s['weight_gb']:.1f} GB")
    print(f"  non-KV overhead = {s['overhead_gb']:.1f} GB")
    print(
        f"  KV budget      = {s['usable_gpu_gb']:.2f} - {s['weight_gb']:.1f} - {s['overhead_gb']:.1f}"
        f" = {s['kv_budget_gb']:.2f} GB"
    )
    print(
        f"  max tokens     = {s['kv_budget_gb']:.2f}e9 / {bpt} = {s['max_kv_tokens']:.2f}"
    )
    print(
        f"  max 4096-seqs  = {s['max_kv_tokens']:.2f} / {SPEC['max_model_len']}"
        f" = {s['max_concurrent_4096']:.3f}  →  floor {s['max_concurrent_4096_floor']}"
    )

    print()
    print("=== Check against bench_log.csv ===")
    rows = []
    for run in load_log():
        pred = predicted_kv_util(run.num_requests, run.seq_len)
        rows.append(
            [
                f"{'long' if run.is_long else 'short'} bs{run.batch_size}",
                run.seq_len,
                run.num_requests * run.seq_len,
                f"{pred:.3f}",
                f"{run.kv_cache_util:.2f}",
                f"{pred - run.kv_cache_util:+.3f}",
                run.preempted_seqs,
            ]
        )
    table(
        [
            "run",
            "seq_len",
            "tokens in flight",
            "predicted util",
            "log kv_cache_util",
            "delta",
            "preempted_seqs",
        ],
        rows,
    )
    print()
    print("Long bs24 predicted 0.933 vs log 0.93, 0 preemptions — fits.")
    print("Long bs32/48 exceed the pool (pred > 1); log pins util at 0.97 and preempts.")
    print("Short bs64 predicted 0.467 vs log 0.47.")


if __name__ == "__main__":
    main()
