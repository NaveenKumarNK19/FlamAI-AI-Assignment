#!/usr/bin/env python3
"""
resource_envelope.py — Part C back-of-envelope

Reviewer hours, labeled-pair budget, GPU idle time vs SFT/rewrite cost,
and L4 rewriter latency from Part B batch-1 ITL.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- constraints (brief) ---
REVIEWER_H_PER_WEEK = 10
WEEKS_TO_LAUNCH = 3
WEEKS_OF_LABELING = 2  # week 3 is launch review; freeze labels earlier
A100_WEEKS = 2
A100_H_PER_DAY = 24  # exclusive reservation, upper bound
LANGUAGES = ("hi", "kn", "ta", "te", "bn", "mr")
REVIEWABLE = ("hi", "kn")

# --- labeling throughput ---
# Pairwise: baseline vs candidate, plus 1-5 casualness and meaning-ok flag.
SECONDS_PER_LABELED_ITEM = 120  # 2 minutes, native-speaker, not skim

# --- generation / train planning assumptions (unbenchmarked) ---
SELF_REWRITE_SEC_PER_PAIR = 2.0
LORA_SFT_HOURS_4B = 6.0
REWRITE_TRAIN_HOURS_1B = 8.0

# --- serving (Part B: L4 batch-1 ITL ~43 ms for 4B) ---
L4_ITL_S_4B = 0.04348
TYPICAL_REPLY_TOKENS = 150
LOADED_L4_ITL_S_4B = 0.09607  # Part B long-context batch-24 row
REWRITER_ITL_FACTOR = 0.35  # unmeasured 1B planning estimate


def main():
    configure_stdio()
    label_hours = REVIEWER_H_PER_WEEK * WEEKS_OF_LABELING
    items_per_hour = 3600 / SECONDS_PER_LABELED_ITEM
    labeled_budget = int(label_hours * items_per_hour)
    per_reviewable_lang = labeled_budget // len(REVIEWABLE)
    eval_holdout = 100
    pilot_total = 60
    remaining_after_eval = labeled_budget - 2 * eval_holdout - pilot_total
    sft_seed = max(remaining_after_eval // len(REVIEWABLE), 0)

    gpu_hours = A100_WEEKS * 7 * A100_H_PER_DAY
    gen_10k_hours = 10_000 * SELF_REWRITE_SEC_PER_PAIR / 3600

    extra_s_4b_speed = TYPICAL_REPLY_TOKENS * L4_ITL_S_4B
    extra_s_4b_loaded = TYPICAL_REPLY_TOKENS * LOADED_L4_ITL_S_4B
    extra_s_1b = TYPICAL_REPLY_TOKENS * L4_ITL_S_4B * REWRITER_ITL_FACTOR

    print("=== Reviewer envelope ===")
    print(f"  languages wanted:     {len(LANGUAGES)} {LANGUAGES}")
    print(f"  reviewer can judge:   {REVIEWABLE} only")
    print(f"  hours before freeze:  {REVIEWER_H_PER_WEEK} h/wk × {WEEKS_OF_LABELING} wk = {label_hours} h")
    print(f"  cadence:              {SECONDS_PER_LABELED_ITEM}s/item → {items_per_hour:.0f} items/h")
    print(f"  labeled budget:       {labeled_budget} items total (~{per_reviewable_lang} per HI/KN)")
    print(f"  day-1 pilot:          {pilot_total} judgments total (30 HI + 30 KN)")
    print(f"  holdout eval:         {eval_holdout} HI + {eval_holdout} KN")
    print(f"  leftover after both:  ~{sft_seed} per HI/KN (not 50k synthetic SFT)")
    print(f"  TA/TE/BN/MR gold:     0")

    print()
    print("=== A100-80GB envelope (2 weeks exclusive) ===")
    print(f"  upper bound:          {gpu_hours} GPU-h")
    print(f"  10k self-rewrites:    {gen_10k_hours:.1f} GPU-h")
    print(f"  4B LoRA SFT:          ~{LORA_SFT_HOURS_4B} GPU-h")
    print(f"  1B rewriter train:    ~{REWRITE_TRAIN_HOURS_1B} GPU-h")
    print("  note:                 these three runtimes are planning assumptions, not benchmarks")
    print("  binding constraint:   reviewer coverage, not GPU")

    print()
    print("=== Path (b) serving tax on the Part B L4 ===")
    print(
        f"  extra decode @4B ITL: {extra_s_4b_speed:.1f}–{extra_s_4b_loaded:.1f} s/reply "
        "(unloaded batch-1 to loaded long-context ITL)"
    )
    print(f"  extra decode @1B est: {extra_s_1b:.1f} s/reply unloaded (unmeasured)")
    print("  Part B already memory-bound on this L4; a second model is a product tax forever")

    print()
    print("=== Success threshold (HI+KN holdout, n=100 each) ===")
    print("  mean casualness ≥ 3.5 / 5")
    print("  ≥ 70% of items rated ≥ 3")
    print("  meaning-preserved ≥ 95%")
    print("  baseline expected ~2.0–2.5 (textbook)")

    print()
    print("=== Kill (c) by end of week 1 ===")
    print("  if Δ mean casualness < 0.8 vs baseline, or meaning-preserved < 95%")
    print("  then stop pure prompt-eng and start HI+KN LoRA on reviewer-filtered self-rewrites")


if __name__ == "__main__":
    main()
