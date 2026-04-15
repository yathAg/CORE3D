#!/usr/bin/env python3
"""Slice ScanRefer failures from an enriched preds_val.jsonl.

Usage:
    python scripts/analyze_scanrefer_failures.py <preds_val.jsonl> [more.jsonl ...]

Each input file is analyzed independently, then a side-by-side comparison is
printed at the end. Requires the v2 schema (query_text, subset, target_label,
pred_label, target_rank_by_size, num_same_class, target_size, pred_size).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

KEYWORDS = {
    "spatial":  r"\b(left|right|front|behind|above|below|next to|between|corner|middle|center|near|far|closest|farthest|nearest)\b",
    "color":    r"\b(black|white|gray|grey|red|orange|yellow|green|blue|purple|pink|brown|dark|light|colou?red)\b",
    "size":     r"\b(small|large|big|tall|short|tiny|huge|long|wide|narrow)\b",
    "material": r"\b(wooden|wood|metal|leather|fabric|plastic|glass|stone|marble)\b",
    "ordinal":  r"\b(first|second|third|fourth|last|middle)\b",
}
KEYWORD_RE = {k: re.compile(p, re.IGNORECASE) for k, p in KEYWORDS.items()}

CARD_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10**9, "6+")]
RANK_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10"), (11, 10**9, ">10")]


def bucket(value: int, buckets) -> str:
    if value is None:
        return "?"
    for lo, hi, label in buckets:
        if lo <= value <= hi:
            return label
    return "?"


def load(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def acc_table(rows: List[Dict], key_fn, ordering=None) -> List[Tuple[str, int, int, float]]:
    """Return [(bucket, n_total, n_correct, acc)] sorted by ordering or by -n_total."""
    correct = Counter()
    total = Counter()
    for r in rows:
        b = key_fn(r)
        if b is None:
            continue
        total[b] += 1
        gt = r.get("gt_object_id_1indexed")
        pred = r.get("prediction")
        if gt is not None and pred is not None and int(pred) == int(gt):
            correct[b] += 1
    items = list(total.keys())
    if ordering:
        items = [b for b in ordering if b in total] + [b for b in items if b not in ordering]
    else:
        items.sort(key=lambda b: -total[b])
    return [(b, total[b], correct[b], correct[b] / total[b] if total[b] else 0.0) for b in items]


def fmt_table(title: str, rows: List[Tuple[str, int, int, float]]):
    print(f"\n## {title}")
    print(f"  {'bucket':<14} {'N':>6} {'correct':>8} {'acc':>7}")
    for b, n, c, a in rows:
        print(f"  {b:<14} {n:>6d} {c:>8d} {a:>7.3f}")


def query_keyword_buckets(query: str) -> List[str]:
    if not query:
        return []
    hits = [k for k, rx in KEYWORD_RE.items() if rx.search(query)]
    return hits or ["other"]


def analyze_one(path: Path):
    rows = load(path)
    n = len(rows)
    print(f"\n========================================================================")
    print(f"# {path}")
    print(f"  n_queries = {n}")
    if not n:
        return rows

    schema_keys = set(rows[0].keys())
    enriched = {"query_text", "subset", "target_label", "pred_label",
                "target_rank_by_size", "num_same_class"} <= schema_keys
    if not enriched:
        print(f"  ! schema is OLD (missing enriched fields). Re-run with current llm/benchmarks.py.")
        print(f"  fields present: {sorted(schema_keys)}")
        return rows

    correct_total = sum(
        1 for r in rows
        if r.get("prediction") is not None
        and r.get("gt_object_id_1indexed") is not None
        and int(r["prediction"]) == int(r["gt_object_id_1indexed"])
    )
    print(f"  overall_accuracy = {correct_total / n:.4f}  ({correct_total}/{n})")
    parse_fail = sum(1 for r in rows if r.get("parse_fail"))
    target_missing = sum(1 for r in rows if not r.get("target_in_graph"))
    print(f"  parse_fail = {parse_fail}   target_missing_from_graph = {target_missing}")

    fmt_table("Subset", acc_table(rows, lambda r: r.get("subset"),
                                  ordering=["unique", "multiple", "unknown"]))

    fmt_table("Same-class cardinality (num_same_class)",
              acc_table(rows, lambda r: bucket(r.get("num_same_class"), CARD_BUCKETS),
                        ordering=[lbl for _, _, lbl in CARD_BUCKETS]))

    fmt_table("Target rank by size",
              acc_table(rows, lambda r: bucket(r.get("target_rank_by_size"), RANK_BUCKETS),
                        ordering=[lbl for _, _, lbl in RANK_BUCKETS]))

    # Query keyword buckets — a query can hit multiple, count separately
    kw_correct = Counter()
    kw_total = Counter()
    for r in rows:
        for k in query_keyword_buckets(r.get("query_text", "")):
            kw_total[k] += 1
            if r.get("prediction") is not None and r.get("gt_object_id_1indexed") is not None \
                    and int(r["prediction"]) == int(r["gt_object_id_1indexed"]):
                kw_correct[k] += 1
    print(f"\n## Query keyword buckets (queries can match multiple)")
    print(f"  {'bucket':<14} {'N':>6} {'correct':>8} {'acc':>7}")
    for k in sorted(kw_total, key=lambda k: -kw_total[k]):
        print(f"  {k:<14} {kw_total[k]:>6d} {kw_correct[k]:>8d} {kw_correct[k]/kw_total[k]:>7.3f}")

    # Confusion matrix among wrong picks (Multiple subset only — Unique is trivially right-class)
    confusion = Counter()
    wrong_multi = 0
    for r in rows:
        if r.get("subset") != "multiple":
            continue
        pred = r.get("prediction")
        gt = r.get("gt_object_id_1indexed")
        if pred is None or gt is None or int(pred) == int(gt):
            continue
        wrong_multi += 1
        confusion[(r.get("target_label"), r.get("pred_label"))] += 1
    print(f"\n## Top-30 (target_label, pred_label) confusions among Multiple-subset wrong picks (n_wrong={wrong_multi})")
    print(f"  {'count':>6}  {'target → pred'}")
    for (tl, pl), c in confusion.most_common(30):
        print(f"  {c:>6d}  {tl} → {pl}")

    # Position bias on wrong picks — distribution of pred_rank_by_size
    # pred_rank_by_size isn't directly logged, but we can use pred's position
    # in the ordering implied by size_lookup. We don't have that here without
    # the graph; approximate using whether pred matches one of N largest by
    # comparing pred_size to target_size's neighborhood. Skip for now —
    # would require graph load.

    # Wrong-pick label-class match (LLM picked SAME class as target?)
    same_class_wrong = sum(
        1 for r in rows
        if r.get("subset") == "multiple"
        and r.get("prediction") is not None
        and r.get("gt_object_id_1indexed") is not None
        and int(r["prediction"]) != int(r["gt_object_id_1indexed"])
        and r.get("pred_label") is not None
        and r.get("pred_label") == r.get("target_label")
    )
    print(f"\n## Multiple-subset wrong picks where LLM chose SAME class as target: {same_class_wrong}/{wrong_multi}")
    print(f"   (high ratio = LLM disambiguates within class poorly; low ratio = wrong class entirely)")

    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            sys.exit(1)
    for p in paths:
        analyze_one(p)


if __name__ == "__main__":
    main()
