#!/usr/bin/env python3
"""Compute mean F1@0.25/0.50 for Multi3DRefer predictions (JSONL).

Reads preds_<split>.jsonl from a run directory, loads per-scene graphs to
retrieve pred_alignment IoU values (for OneFormer graphs) and evaluates
set-level precision/recall/F1 for each query:
  - TP: predicted id in GT set with IoU >= threshold
  - FP: predicted id not matched (not in GT or IoU < threshold)
  - FN: GT ids not matched by any TP
Empty-GT queries are treated as incorrect by default (FN += 1 if no preds,
FP += len(preds) if preds exist). For zt_* types, an empty prediction is
treated as correct (F1=1), mirroring 3DGraphLLM behavior.
"""

import argparse
import csv
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _find_run_params(run_dir: str, split: str) -> Optional[Dict]:
    candidates = [
        osp.join(run_dir, f"all_run_params_{split}.json"),
        osp.join(run_dir, "all_run_params.json"),
    ]
    for p in candidates:
        if osp.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _resolve_defaults() -> Dict[str, str]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    return {
        "graph_root": str(repo_root / "results/graphs/scannet200_oneformer/scenes"),
    }


def _normalize_id_list(pred) -> List[int]:
    if pred is None:
        return []
    if isinstance(pred, dict):
        if "id" in pred:
            return _normalize_id_list(pred.get("id"))
        if "prediction" in pred:
            return _normalize_id_list(pred.get("prediction"))
        return []
    if isinstance(pred, (int, float, str)):
        try:
            return [int(pred)]
        except Exception:
            return []
    if isinstance(pred, list):
        out: List[int] = []
        for p in pred:
            out.extend(_normalize_id_list(p))
        return out
    return []


def _extract_gt_ids(rec: Dict) -> List[int]:
    for key in (
        "gt_object_ids_eval",
        "gt_object_ids_1idx",
        "gt_object_id_1indexed",
        "gt_object_ids",
        "gt_object_id",
    ):
        if key in rec and rec[key] is not None:
            return _normalize_id_list(rec[key])
    return []


def _load_predictions(path: str):
    by_scene: Dict[str, List[Dict]] = defaultdict(list)
    total = 0
    parse_fails = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scene_id = rec.get("scene_id") or rec.get("scene")
            if scene_id is None:
                continue
            parse_fail = bool(rec.get("parse_fail", False))
            skipped = bool(rec.get("skipped", False))
            pred_ids = _normalize_id_list(rec.get("prediction"))
            if parse_fail:
                parse_fails += 1
            entry = {
                "scene_id": scene_id,
                "idx": rec.get("idx"),
                "gt_ids": _extract_gt_ids(rec),
                "pred_ids": pred_ids,
                "parse_fail": parse_fail,
                "skipped": skipped,
                "question_type": rec.get("question_type") or rec.get("eval_type") or "unknown",
            }
            by_scene[scene_id].append(entry)
            total += 1
    return by_scene, total, parse_fails


def _is_gt_graph(graph: Optional[Dict]) -> bool:
    if not graph:
        return False
    if graph.get("prediction_source") == "oneformer3d":
        return False
    if graph.get("pred_alignment") is not None:
        return False
    return True


def _load_graph_iou_map(graph: Dict) -> Dict[int, float]:
    mapping_raw = graph.get("pred_alignment") or {}
    iou_map: Dict[int, float] = {}
    for k, v in mapping_raw.items():
        try:
            gid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        iou = v.get("iou")
        try:
            iou_map[gid] = float(iou) if iou is not None else 0.0
        except Exception:
            iou_map[gid] = 0.0
    return iou_map


def _load_valid_ids(graph: Dict) -> Optional[set]:
    objs = graph.get("objects")
    if not isinstance(objs, dict):
        return None
    ids = set()
    for k in objs.keys():
        try:
            ids.add(int(k))
        except Exception:
            continue
    return ids


def _score_counts(
    gt_ids: List[int],
    pred_ids: List[int],
    iou_map: Dict[int, float],
    threshold: float,
    use_gt_masks: bool,
    empty_gt_behavior: str,
) -> Tuple[int, int, int]:
    gt_set = {int(x) for x in gt_ids if x is not None}
    pred_set = {int(x) for x in pred_ids if x is not None}

    if not gt_set:
        if empty_gt_behavior == "skip":
            return 0, 0, 0
        if not pred_set:
            return 0, 0, 1
        return 0, len(pred_set), 0

    tp = 0
    fp = 0
    for pid in pred_set:
        if pid in gt_set:
            iou = 1.0 if use_gt_masks else iou_map.get(pid, 0.0)
            if iou >= threshold:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = len(gt_set) - tp
    return tp, fp, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def _init_stats() -> Dict[str, float]:
    return {
        "evaluated": 0,
        "skipped": 0,
        "parse_fail": 0,
        "empty_gt": 0,
        "f1_25_sum": 0.0,
        "f1_50_sum": 0.0,
    }


def _update_stats(stat: Dict[str, float], *, skipped: bool = False, parse_fail: bool = False, empty_gt: bool = False,
                 f1_25: float = 0.0, f1_50: float = 0.0) -> None:
    if skipped:
        stat["skipped"] += 1
        return
    stat["evaluated"] += 1
    if parse_fail:
        stat["parse_fail"] += 1
    if empty_gt:
        stat["empty_gt"] += 1
    stat["f1_25_sum"] += f1_25
    stat["f1_50_sum"] += f1_50


def _finalize_stats(stat: Dict[str, float]) -> Dict[str, float]:
    evaluated = int(stat["evaluated"])
    f1_25_macro = (stat["f1_25_sum"] / evaluated) if evaluated else 0.0
    f1_50_macro = (stat["f1_50_sum"] / evaluated) if evaluated else 0.0
    return {
        "evaluated": evaluated,
        "skipped": int(stat["skipped"]),
        "parse_fail": int(stat["parse_fail"]),
        "empty_gt": int(stat["empty_gt"]),
        "f1_25_mean": f1_25_macro,
        "f1_50_mean": f1_50_macro,
    }


def main() -> None:
    defaults = _resolve_defaults()
    ap = argparse.ArgumentParser(description="F1 evaluation for Multi3DRefer JSONL predictions.")
    ap.add_argument("--run-dir", required=True, help="Run folder containing preds_<split>.jsonl")
    ap.add_argument("--split", required=True, help="Split name (train/val/all)")
    ap.add_argument("--graph-root", default=None, help="Override graph root for scene JSONs")
    ap.add_argument(
        "--empty-gt",
        default="incorrect",
        choices=["incorrect", "skip"],
        help="How to handle queries with empty GT sets (default: incorrect)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    split = args.split
    preds_path = osp.join(run_dir, f"preds_{split}.jsonl")
    if not osp.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    run_params = _find_run_params(run_dir, split)
    graph_root = args.graph_root or (
        run_params.get("graph_root_resolved") if run_params else None
    ) or defaults["graph_root"]

    entries_by_scene, total_entries, parse_fails = _load_predictions(preds_path)

    overall = _init_stats()
    by_qtype: Dict[str, Dict[str, float]] = {}

    zt_types = {"zt_w_d", "zt_wo_d"}
    for scene_id, entries in entries_by_scene.items():
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        graph = None
        if osp.exists(graph_path):
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
            except Exception:
                graph = None
        if graph is None:
            for e in entries:
                qt = e.get("question_type") or "unknown"
                by_qtype.setdefault(qt, _init_stats())
                _update_stats(overall, skipped=True)
                _update_stats(by_qtype[qt], skipped=True)
            continue

        use_gt_masks = _is_gt_graph(graph)
        iou_map = {} if use_gt_masks else _load_graph_iou_map(graph)

        for e in entries:
            qt = e.get("question_type") or "unknown"
            by_qtype.setdefault(qt, _init_stats())

            if e.get("skipped"):
                _update_stats(overall, skipped=True)
                _update_stats(by_qtype[qt], skipped=True)
                continue

            gt_ids = e.get("gt_ids") or []
            pred_ids = [] if e.get("parse_fail") else (e.get("pred_ids") or [])
            empty_gt = len(gt_ids) == 0

            if qt in zt_types:
                if len(pred_ids) == 0:
                    f1_25 = 1.0
                    f1_50 = 1.0
                else:
                    f1_25 = 0.0
                    f1_50 = 0.0
            else:
                if empty_gt and args.empty_gt == "skip":
                    _update_stats(overall, skipped=True)
                    _update_stats(by_qtype[qt], skipped=True)
                    continue

                tp25, fp25, fn25 = _score_counts(
                    gt_ids,
                    pred_ids,
                    iou_map,
                    0.25,
                    use_gt_masks,
                    args.empty_gt,
                )
                tp50, fp50, fn50 = _score_counts(
                    gt_ids,
                    pred_ids,
                    iou_map,
                    0.50,
                    use_gt_masks,
                    args.empty_gt,
                )

                f1_25 = _f1_from_counts(tp25, fp25, fn25)
                f1_50 = _f1_from_counts(tp50, fp50, fn50)

            _update_stats(
                overall,
                parse_fail=e.get("parse_fail", False),
                empty_gt=empty_gt,
                f1_25=f1_25,
                f1_50=f1_50,
            )
            _update_stats(
                by_qtype[qt],
                parse_fail=e.get("parse_fail", False),
                empty_gt=empty_gt,
                f1_25=f1_25,
                f1_50=f1_50,
            )

    overall_fmt = _finalize_stats(overall)
    out_csv = osp.join(run_dir, f"metric_f1_{split}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "evaluated",
                "skipped",
                "parse_fail",
                "empty_gt",
                "f1_mean@0.25",
                "f1_mean@0.50",
            ]
        )
        writer.writerow(
            [
                "overall",
                overall_fmt["evaluated"],
                overall_fmt["skipped"],
                overall_fmt["parse_fail"],
                overall_fmt["empty_gt"],
                f"{overall_fmt['f1_25_mean']:.4f}",
                f"{overall_fmt['f1_50_mean']:.4f}",
            ]
        )
        for qt in sorted(by_qtype.keys()):
            fmt = _finalize_stats(by_qtype[qt])
            writer.writerow(
                [
                    qt,
                    fmt["evaluated"],
                    fmt["skipped"],
                    fmt["parse_fail"],
                    fmt["empty_gt"],
                    f"{fmt['f1_25_mean']:.4f}",
                    f"{fmt['f1_50_mean']:.4f}",
                ]
            )

    print(
        f"Saved F1 summary to {out_csv}. "
        f"F1@0.25(mean)={overall_fmt['f1_25_mean']:.4f}, "
        f"F1@0.50(mean)={overall_fmt['f1_50_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
