#!/usr/bin/env python3
"""FLM-4B / L4 capacity math and bench_log derived metrics (Part B)."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Submission layout: CSV/spec live in partB/bench/. (In the intern workspace they sat at starter_kit/bench/.)
BENCH_DIR = ROOT / "bench" if (ROOT / "bench" / "bench_log.csv").exists() else ROOT.parent / "bench"
LOG_PATH = BENCH_DIR / "bench_log.csv"

# Values copied from bench/model_spec.md — arithmetic lives here, not in prose.
SPEC = {
    "params": 4.2e9,
    "layers": 28,
    "d_model": 3072,
    "q_heads": 24,
    "kv_heads": 8,
    "head_dim": 128,
    "weight_bytes": 2,  # fp16
    "kv_bytes": 2,  # fp16
    "gpu_gb": 24.0,
    "gpu_mem_util": 0.92,
    "overhead_gb": 1.6,
    "max_model_len": 4096,
    "bandwidth_gbs": 300.0,
}


def configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def kv_bytes_per_token() -> int:
    """Exact KV-cache bytes for one token (all layers, K and V)."""
    s = SPEC
    return s["layers"] * s["kv_heads"] * s["head_dim"] * 2 * s["kv_bytes"]


def weight_gb() -> float:
    return SPEC["params"] * SPEC["weight_bytes"] / 1e9


def usable_gpu_gb() -> float:
    return SPEC["gpu_gb"] * SPEC["gpu_mem_util"]


def kv_budget_gb() -> float:
    return usable_gpu_gb() - weight_gb() - SPEC["overhead_gb"]


def kv_budget_bytes() -> float:
    return kv_budget_gb() * 1e9


def max_kv_tokens() -> float:
    return kv_budget_bytes() / kv_bytes_per_token()


def max_concurrent_full_seqs() -> float:
    return max_kv_tokens() / SPEC["max_model_len"]


def predicted_kv_util(n_seqs: int, seq_len: int) -> float:
    """Fraction of the KV pool occupied if n_seqs each hold seq_len tokens."""
    return (n_seqs * seq_len * kv_bytes_per_token()) / kv_budget_bytes()


@dataclass
class Run:
    batch_size: int
    prompt_len: int
    gen_len: int
    num_requests: int
    wall_clock_s: float
    reported_tok_s: float
    ttft_ms_p50: float
    itl_ms_p50: float
    e2e_ms_p95: float
    preempted_seqs: int
    kv_cache_util: float

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.gen_len

    @property
    def is_long(self) -> bool:
        return self.prompt_len == 3584

    @property
    def reconstructed_reported(self) -> float:
        """What reported_tok_s is: (prompt+gen)*n / wall."""
        return self.seq_len * self.num_requests / self.wall_clock_s

    @property
    def gen_goodput(self) -> float:
        """Honest generation tok/s from wall clock."""
        return self.gen_len * self.num_requests / self.wall_clock_s

    @property
    def gen_goodput_from_reported(self) -> float:
        """Same goodput via the contaminated column."""
        return self.reported_tok_s * self.gen_len / self.seq_len

    @property
    def decode_tok_s_from_itl(self) -> float:
        """Independent decode-rate view: concurrent seqs / median ITL."""
        return self.batch_size / (self.itl_ms_p50 / 1000.0)

    @property
    def predicted_kv_util(self) -> float:
        return predicted_kv_util(self.num_requests, self.seq_len)


def load_log(path: Path = LOG_PATH) -> list[Run]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(
                Run(
                    batch_size=int(raw["batch_size"]),
                    prompt_len=int(raw["prompt_len"]),
                    gen_len=int(raw["gen_len"]),
                    num_requests=int(raw["num_requests"]),
                    wall_clock_s=float(raw["wall_clock_s"]),
                    reported_tok_s=float(raw["reported_tok_s"]),
                    ttft_ms_p50=float(raw["ttft_ms_p50"]),
                    itl_ms_p50=float(raw["itl_ms_p50"]),
                    e2e_ms_p95=float(raw["e2e_ms_p95"]),
                    preempted_seqs=int(raw["preempted_seqs"]),
                    kv_cache_util=float(raw["kv_cache_util"]),
                )
            )
    return rows


def long_runs(rows: list[Run] | None = None) -> list[Run]:
    rows = rows if rows is not None else load_log()
    return [r for r in rows if r.is_long]


def short_runs(rows: list[Run] | None = None) -> list[Run]:
    rows = rows if rows is not None else load_log()
    return [r for r in rows if not r.is_long]


def find_run(batch: int, prompt: int, rows: list[Run] | None = None) -> Run:
    rows = rows if rows is not None else load_log()
    for r in rows:
        if r.batch_size == batch and r.prompt_len == prompt:
            return r
    raise KeyError(f"no row batch={batch} prompt={prompt}")


def b1_summary() -> dict:
    bpt = kv_bytes_per_token()
    return {
        "kv_bytes_per_token": bpt,
        "kv_kib_per_token": bpt / 1024,
        "weight_gb": weight_gb(),
        "usable_gpu_gb": usable_gpu_gb(),
        "overhead_gb": SPEC["overhead_gb"],
        "kv_budget_gb": kv_budget_gb(),
        "max_kv_tokens": max_kv_tokens(),
        "max_concurrent_4096": max_concurrent_full_seqs(),
        "max_concurrent_4096_floor": int(max_concurrent_full_seqs()),
        "head_dim_check": SPEC["d_model"] / SPEC["q_heads"],
    }
