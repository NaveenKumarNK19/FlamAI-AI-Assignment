#!/usr/bin/env python3
"""
analyze_bench.py — Part B2 / B3

Reconstruct reported_tok_s, locate the long-context throughput anomaly,
and compute honest generation goodput via two equivalent derivations.
"""

from __future__ import annotations

from bench_metrics import configure_stdio, find_run, load_log, long_runs, short_runs


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
    log = load_log()

    print("=== reported_tok_s is (prompt+gen)*n / wall ===")
    recon_rows = []
    for r in log:
        recon_rows.append(
            [
                f"{'L' if r.is_long else 'S'}{r.batch_size}",
                f"{r.reported_tok_s:.1f}",
                f"{r.reconstructed_reported:.1f}",
                f"{r.reported_tok_s - r.reconstructed_reported:+.2f}",
                f"{r.gen_goodput:.1f}",
            ]
        )
    table(
        ["run", "reported_tok_s", "(p+g)*n/wall", "delta", "gen goodput"],
        recon_rows,
    )

    print()
    print("=== B2  Long-context sweep (prompt=3584, gen=512) ===")
    long = long_runs(log)
    long_rows = []
    prev = None
    for r in long:
        naive = None if prev is None else prev.reported_tok_s * (r.batch_size / prev.batch_size)
        long_rows.append(
            [
                r.batch_size,
                f"{r.reported_tok_s:.1f}",
                f"{naive:.1f}" if naive else "—",
                f"{r.gen_goodput:.1f}",
                f"{r.ttft_ms_p50:.1f}",
                f"{r.itl_ms_p50:.1f}",
                f"{r.e2e_ms_p95/1000:.1f}s",
                f"{r.kv_cache_util:.2f}",
                r.preempted_seqs,
                f"{r.wall_clock_s:.2f}",
            ]
        )
        prev = r
    table(
        [
            "bs",
            "reported tok/s",
            "naive scale from prev",
            "gen tok/s",
            "ttft p50",
            "itl p50",
            "e2e p95",
            "kv util",
            "preempt",
            "wall s",
        ],
        long_rows,
    )
    peak = max(long, key=lambda r: r.reported_tok_s)
    worst = long[-1]
    print()
    print(
        f"Anomaly: peak at bs{peak.batch_size} ({peak.reported_tok_s:.1f} tok/s, "
        f"0 preempt, kv={peak.kv_cache_util:.2f}), then "
        f"bs32={find_run(32, 3584, log).reported_tok_s:.1f} and "
        f"bs48={worst.reported_tok_s:.1f} with preempt 7 and 23."
    )
    print(
        "Naive batch-24 → 48: "
        f"{peak.reported_tok_s:.0f} * 48/24 = {peak.reported_tok_s * 2:.0f} tok/s. "
        f"Log: {worst.reported_tok_s:.1f} ({100*(worst.reported_tok_s/peak.reported_tok_s - 1):+.0f}%)."
    )
    print(
        "Config: --max-num-seqs 24 for 4k traffic. From the measured bs24 row, "
        "the active set should use ~0.93 KV and avoid preemption at ~201 generated "
        "tok/s. A 48-request burst will queue 24 requests, so its end-to-end p95 "
        "must be load-tested; the bs24 value of 69s is not a valid bs48 prediction."
    )

    print()
    print("=== B3  Honest goodput, batch-24 long row ===")
    r24 = find_run(24, 3584, log)
    r16s = find_run(16, 512, log)
    r16l = find_run(16, 3584, log)
    r48 = find_run(48, 3584, log)
    way1 = r24.gen_goodput
    way2 = r24.gen_goodput_from_reported
    itl = r24.decode_tok_s_from_itl
    print(f"  way 1  num_requests * gen_len / wall_clock_s")
    print(f"         {r24.num_requests} * {r24.gen_len} / {r24.wall_clock_s} = {way1:.2f} gen tok/s")
    print(f"  way 2  reported_tok_s * gen_len / (prompt_len + gen_len)")
    print(
        f"         {r24.reported_tok_s} * {r24.gen_len} / {r24.seq_len} = {way2:.2f} gen tok/s"
    )
    print(f"  (ITL view, not required for the identity: {r24.batch_size} / {r24.itl_ms_p50/1000:.5f}s = {itl:.1f} decode tok/s)")
    print()
    print("v0 compared bs16 long vs short reported_tok_s (1311 vs 883). End-to-end generation goodput:")
    print(f"  short bs16: {r16s.gen_goodput:.1f}   long bs16: {r16l.gen_goodput:.1f}")
    print(
        f"  ITL also worsens from {r16s.itl_ms_p50:.2f} to {r16l.itl_ms_p50:.2f} ms/token; "
        "the rows have different generation lengths, so goodput alone is not a controlled decode comparison."
    )
    print(
        f"  v0 batch-48 ~3200 from 1600*2. Log bs48 reported={r48.reported_tok_s:.1f}, "
        f"gen={r48.gen_goodput:.1f}, preempt={r48.preempted_seqs}."
    )


if __name__ == "__main__":
    main()
