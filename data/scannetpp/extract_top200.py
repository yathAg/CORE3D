#!/usr/bin/env python3
"""Build top200 semantic + instance allowlists for ScanNet++.

Counts mapped semantic labels across all scenes. Mapping uses
metadata/semantic_benchmark/map_benchmark.csv (semantic_map_to).
If no mapping exists, the raw label is used. Generic "object" labels
are ignored. Output:
  - topK_semantic.txt: stuff (if present) first, then things
  - topK_instance.txt: things only (stuff removed)
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Set


def load_semantic_map(map_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map file: {map_path}")
    with map_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("class") or "").strip().lower()
            dst = (row.get("semantic_map_to") or "").strip().lower()
            if src and dst:
                mapping[src] = dst
    return mapping


def iter_segments(scene_root: Path) -> Iterable[Path]:
    return scene_root.glob("*/scans/segments_anno.json")


_LABEL_RE = re.compile(br'"label"\s*:\s*"([^"]+)"')
_IGNORE_LABELS = {"remove", "split"}
_STUFF_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "cieling",
    "tiled wall",
    "bathroom wall",
    "shower wall",
}


def _extract_labels_fast(seg_path: Path) -> List[str]:
    try:
        data = seg_path.read_bytes()
    except Exception:
        return []
    labels = []
    for match in _LABEL_RE.finditer(data):
        try:
            labels.append(match.group(1).decode("utf-8", errors="ignore").strip().lower())
        except Exception:
            continue
    return labels


def build_counts(
    scene_root: Path,
    sem_map: Dict[str, str],
    workers: int,
    ignore_labels: Set[str],
) -> Counter:
    counts: Counter = Counter()
    seg_paths = list(iter_segments(scene_root))
    if workers <= 1:
        for seg_path in seg_paths:
            for label in _extract_labels_fast(seg_path):
                if not label:
                    continue
                label = sem_map.get(label, label)
                if label in ignore_labels:
                    continue
                counts[label] += 1
        return counts

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_labels_fast, p): p for p in seg_paths}
        for fut in as_completed(futs):
            labels = fut.result() or []
            for label in labels:
                if not label:
                    continue
                label = sem_map.get(label, label)
                if label in ignore_labels:
                    continue
                counts[label] += 1
    return counts


def write_topk(counts: Counter, out_path: Path, k: int, stuff_labels: Set[str]) -> None:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    topk = [name for name, _ in items[:k]]
    if stuff_labels:
        stuff = [name for name in topk if name in stuff_labels]
        things = [name for name in topk if name not in stuff_labels]
        ordered = stuff + things
    else:
        ordered = topk
    out_path.write_text("\n".join(ordered) + "\n")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract top-K semantic labels for ScanNet++")
    parser.add_argument("--data-root", type=Path, default=here, help="Path to data/scannetpp root")
    parser.add_argument("--k", type=int, default=200, help="Number of classes to export")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for file parsing")
    parser.add_argument(
        "--semantic-out",
        type=Path,
        default=None,
        help="Output path for semantic allowlist (default: topK_semantic.txt)",
    )
    parser.add_argument(
        "--instance-out",
        type=Path,
        default=None,
        help="Output path for instance allowlist (default: topK_instance.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    scene_root = data_root / "data"
    meta_root = data_root / "metadata"
    if not scene_root.exists():
        raise FileNotFoundError(f"Missing scenes root: {scene_root}")
    if not meta_root.exists():
        raise FileNotFoundError(f"Missing metadata root: {meta_root}")

    map_path = meta_root / "semantic_benchmark" / "map_benchmark.csv"
    semantic_out = args.semantic_out or (meta_root / "semantic_benchmark" /
                                         f"top{args.k}_semantic.txt")
    instance_out = args.instance_out or (meta_root / "semantic_benchmark" /
                                         f"top{args.k}_instance.txt")
    sem_map = load_semantic_map(map_path)
    counts = build_counts(scene_root, sem_map, args.workers, _IGNORE_LABELS)
    if not counts:
        raise RuntimeError("No labels found; check dataset paths.")
    # semantic list: stuff first, then things
    write_topk(counts, semantic_out, args.k, _STUFF_LABELS)

    # instance list: drop stuff from top-k list, preserve ordering
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    topk = [name for name, _ in items[:args.k]]
    instance_only = [name for name in topk if name not in _STUFF_LABELS]
    instance_out.write_text("\n".join(instance_only) + "\n")

    print(f"Wrote {semantic_out} with {args.k} labels.")
    print(f"Wrote {instance_out} with {len(instance_only)} labels (stuff removed).")


if __name__ == "__main__":
    main()
