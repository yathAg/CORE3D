#!/usr/bin/env python3
"""Compute class-conditional mask IoU accuracy for Reason3D predictions (JSONL).

Reason3D object_id is a semantic class index (ScanNet200). We map it to the
raw ScanNet200 semantic id, build the GT class mask, and compute IoU between
the predicted instance mask and the class mask. Outputs metric_iou_<split>.csv
with totals only. Supports both OneFormer graphs and GT graphs.
"""

import argparse
import csv
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# From Reason3D/lavis/datasets/datasets/scannet200_constants.py (VALID_CLASS_IDS_200)
VALID_CLASS_IDS_200 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
    23, 24, 26, 27, 28, 29, 31, 32, 33, 34,
    35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 54, 55, 56, 57,
    58, 59, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 82, 84, 86, 87, 88, 89, 90, 93, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 110, 112, 115, 116, 118, 120, 121, 122,
    125, 128, 130, 131, 132, 134, 136, 138, 139, 140,
    141, 145, 148, 154, 155, 156, 157, 159, 161, 163,
    165, 166, 168, 169, 170, 177, 180, 185, 188, 191,
    193, 195, 202, 208, 213, 214, 221, 229, 230, 232,
    233, 242, 250, 261, 264, 276, 283, 286, 300, 304,
    312, 323, 325, 331, 342, 356, 370, 392, 395, 399,
    408, 417, 488, 540, 562, 570, 572, 581, 609, 748,
    776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170,
    1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181,
    1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191,
)


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
    oneformer_root = script_path.parents[2]
    return {
        "graph_root": str(oneformer_root / "results/graphs/scannet200_oneformer/scenes"),
        "data_root": str(oneformer_root / "data/scannet200"),
        "pred_root": str(oneformer_root / "results/oneformer3d_1xb4_scannet200"),
    }


def _load_predictions(path: str) -> Dict[str, List[Dict]]:
    by_scene: Dict[str, List[Dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("data_type") not in (None, "scannet"):
                continue
            scene_id = rec.get("scene_id") or rec.get("scene")
            if scene_id is None:
                continue
            gt_raw = rec.get("gt_object_id")
            try:
                gt_id = int(gt_raw) if gt_raw is not None else None
            except Exception:
                gt_id = None
            pred_raw = rec.get("prediction")
            try:
                pred_id = int(pred_raw) if pred_raw is not None else None
            except Exception:
                pred_id = None
            entry = {
                "scene_id": scene_id,
                "gt_class_id": gt_id,
                "pred_id": pred_id,
                "parse_fail": bool(rec.get("parse_fail", False)),
                "skipped": bool(rec.get("skipped", False)),
                "data_type": rec.get("data_type"),
            }
            by_scene[scene_id].append(entry)
    return by_scene


def _load_graph_mapping(path: str) -> Dict[int, int]:
    with open(path) as f:
        graph = json.load(f)
    mapping_raw = graph.get("pred_alignment") or {}
    mapping: Dict[int, int] = {}
    for k, v in mapping_raw.items():
        try:
            gid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        pred_idx = v.get("pred_idx")
        if pred_idx is None:
            continue
        try:
            mapping[gid] = int(pred_idx)
        except Exception:
            continue
    return mapping


def _load_sem_raw(data_root: str, scene_id: str) -> Optional[np.ndarray]:
    sem_path = osp.join(data_root, "semantic_mask", f"{scene_id}.bin")
    if not osp.exists(sem_path):
        return None
    return np.fromfile(sem_path, dtype=np.int64)


def _load_gt_inst(data_root: str, scene_id: str) -> Optional[np.ndarray]:
    inst_path = osp.join(data_root, "instance_mask", f"{scene_id}.bin")
    if not osp.exists(inst_path):
        return None
    return np.fromfile(inst_path, dtype=np.int64)


def _load_pred_masks(pred_root: str, scene_id: str, n_points: int) -> Optional[np.ndarray]:
    pred_path = osp.join(pred_root, f"{scene_id}.pth")
    if not osp.exists(pred_path) or n_points <= 0:
        return None
    pred_obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    inst_masks = np.asarray(pred_obj.get("instance_masks"))
    if inst_masks.size == 0 or inst_masks.ndim != 2:
        return None
    n = min(n_points, inst_masks.shape[1])
    if n == 0:
        return None
    return inst_masks[:, :n]


def _is_gt_graph(graph: Optional[Dict]) -> bool:
    if not graph:
        return False
    if graph.get("id_source") == "gt" or graph.get("prediction_source") == "gt":
        return True
    if graph.get("prediction_source") is not None:
        return False
    pred_alignment = graph.get("pred_alignment")
    return pred_alignment is None


def _compute_scene_stats(
    entries: List[Dict],
    mapping: Dict[int, int],
    sem_raw: Optional[np.ndarray],
    inst_masks: Optional[np.ndarray],
    gt_inst: Optional[np.ndarray],
    thresholds: Tuple[float, float],
    use_gt_masks: bool,
):
    t25, t50 = thresholds
    stats = {
        "total": 0,
        "correct25": 0,
        "correct50": 0,
        "parse_fail": 0,
        "skipped": 0,
        "missing_pred_map": 0,
        "missing_pred_idx": 0,
        "missing_files": 0,
        "invalid_class": 0,
        "iou_sum": 0.0,
    }

    if sem_raw is None or (use_gt_masks and gt_inst is None) or (not use_gt_masks and inst_masks is None):
        stats["missing_files"] += len(entries)
        for e in entries:
            stats["total"] += 1
            if e.get("parse_fail"):
                stats["parse_fail"] += 1
            if e.get("skipped"):
                stats["skipped"] += 1
        return stats

    sem_max = int(sem_raw.max()) if sem_raw.size else 0
    sem_sizes = np.bincount(sem_raw, minlength=sem_max + 1)
    gt_sizes = np.bincount(gt_inst, minlength=int(gt_inst.max()) + 1) if use_gt_masks else None

    pred_cache: Dict[int, Tuple[int, np.ndarray]] = {}
    if use_gt_masks:
        needed_pred_ids = set()
        for e in entries:
            if e.get("parse_fail") or e.get("skipped"):
                continue
            pred_id = e.get("pred_id")
            if pred_id is None:
                continue
            needed_pred_ids.add(pred_id)
        for pred_id in needed_pred_ids:
            if pred_id < 0 or pred_id >= len(gt_sizes):
                continue
            p_mask = gt_inst == pred_id
            p_size = int(p_mask.sum())
            if p_size == 0:
                counts = np.zeros_like(sem_sizes)
            else:
                counts = np.bincount(sem_raw[p_mask], minlength=len(sem_sizes))
            pred_cache[pred_id] = (p_size, counts)
    else:
        needed_pred_idxs = set()
        for e in entries:
            if e.get("parse_fail") or e.get("skipped"):
                continue
            pred_id = e.get("pred_id")
            if pred_id is None:
                continue
            pred_idx = mapping.get(pred_id)
            if pred_idx is None:
                continue
            needed_pred_idxs.add(pred_idx)

        for pred_idx in needed_pred_idxs:
            if pred_idx < 0 or pred_idx >= inst_masks.shape[0]:
                continue
            p_mask = inst_masks[pred_idx].astype(bool)
            p_size = int(p_mask.sum())
            if p_size == 0:
                counts = np.zeros_like(sem_sizes)
            else:
                counts = np.bincount(sem_raw[p_mask], minlength=len(sem_sizes))
            pred_cache[pred_idx] = (p_size, counts)

    for e in entries:
        stats["total"] += 1
        if e.get("parse_fail"):
            stats["parse_fail"] += 1
        if e.get("skipped"):
            stats["skipped"] += 1
        if e.get("parse_fail") or e.get("skipped"):
            continue

        pred_id = e.get("pred_id")
        gt_class_id = e.get("gt_class_id")
        if pred_id is None or gt_class_id is None:
            stats["invalid_class"] += 1
            continue
        if gt_class_id < 0 or gt_class_id >= len(VALID_CLASS_IDS_200):
            stats["invalid_class"] += 1
            continue
        raw_id = VALID_CLASS_IDS_200[gt_class_id]
        if raw_id >= len(sem_sizes):
            stats["invalid_class"] += 1
            continue

        if use_gt_masks:
            if pred_id is None or pred_id < 0 or pred_id >= len(gt_sizes):
                stats["missing_pred_map"] += 1
                continue
            cache = pred_cache.get(pred_id)
            if cache is None:
                stats["missing_pred_idx"] += 1
                continue
        else:
            pred_idx = mapping.get(pred_id)
            if pred_idx is None:
                stats["missing_pred_map"] += 1
                continue
            cache = pred_cache.get(pred_idx)
            if cache is None:
                stats["missing_pred_idx"] += 1
                continue
        p_size, counts = cache
        inter = int(counts[raw_id])
        union = p_size + int(sem_sizes[raw_id]) - inter
        iou = (inter / union) if union > 0 else 0.0
        stats["iou_sum"] += float(iou)

        if iou >= t25:
            stats["correct25"] += 1
        if iou >= t50:
            stats["correct50"] += 1

    return stats


def main():
    defaults = _resolve_defaults()
    ap = argparse.ArgumentParser(description="Mask IoU evaluation for Reason3D JSONL predictions.")
    ap.add_argument("--run-dir", required=True, help="Run folder containing preds_<split>.jsonl")
    ap.add_argument("--split", required=True, help="Split name (train/val/all)")
    ap.add_argument("--graph-root", default=None, help="Override graph root for scene JSONs")
    ap.add_argument("--data-root", default=None, help="Override ScanNet200 data root")
    ap.add_argument("--pred-root", default=None, help="Override OneFormer3D prediction root")
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
    graph_run_args = (run_params or {}).get("graph_run_args", {}) or {}
    if not graph_run_args and graph_root:
        fallback_path = osp.join(osp.dirname(osp.normpath(graph_root)), "run_args.json")
        if osp.exists(fallback_path):
            try:
                with open(fallback_path) as f:
                    graph_run_args = json.load(f) or {}
            except Exception:
                graph_run_args = {}
    data_root = args.data_root or graph_run_args.get("data_root") or defaults["data_root"]
    pred_root = args.pred_root or graph_run_args.get("results_dir") or defaults["pred_root"]

    entries_by_scene = _load_predictions(preds_path)
    totals = {
        "total": 0,
        "correct25": 0,
        "correct50": 0,
        "parse_fail": 0,
        "skipped": 0,
        "missing_pred_map": 0,
        "missing_pred_idx": 0,
        "missing_files": 0,
        "invalid_class": 0,
        "iou_sum": 0.0,
    }

    thresholds = (0.25, 0.50)
    for scene_id, entries in entries_by_scene.items():
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        graph = None
        if osp.exists(graph_path):
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
            except Exception:
                graph = None
        use_gt_masks = _is_gt_graph(graph)
        mapping = _load_graph_mapping(graph_path) if (graph and not use_gt_masks) else {}
        sem_raw = _load_sem_raw(data_root, scene_id)
        if use_gt_masks:
            gt_inst = _load_gt_inst(data_root, scene_id)
            inst_masks = None
        else:
            gt_inst = None
            n_points = len(sem_raw) if sem_raw is not None else 0
            inst_masks = _load_pred_masks(pred_root, scene_id, n_points)
        stats = _compute_scene_stats(
            entries, mapping, sem_raw, inst_masks, gt_inst, thresholds, use_gt_masks
        )
        for k in totals:
            totals[k] += stats.get(k, 0)

    total_eval = totals["total"]
    acc25 = (totals["correct25"] / total_eval) if total_eval else 0.0
    acc50 = (totals["correct50"] / total_eval) if total_eval else 0.0
    miou = (totals["iou_sum"] / total_eval) if total_eval else 0.0

    out_csv = osp.join(run_dir, f"metric_iou_{split}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sceneID",
                "total_queries_evaluated",
                "Correct@0.25",
                "Acc@0.25",
                "Correct@0.50",
                "Acc@0.50",
                "mIoU",
                "parse_fail",
                "skipped",
                "invalid_class",
                "missing_pred_map",
                "missing_pred_idx",
                "missing_files",
            ]
        )
        writer.writerow(
            [
                "total",
                total_eval,
                totals["correct25"],
                f"{acc25:.4f}",
                totals["correct50"],
                f"{acc50:.4f}",
                f"{miou:.4f}",
                totals["parse_fail"],
                totals["skipped"],
                totals["invalid_class"],
                totals["missing_pred_map"],
                totals["missing_pred_idx"],
                totals["missing_files"],
            ]
        )

    print(
        f"Saved IoU summary to {out_csv}. Acc@0.25={acc25:.4f}, Acc@0.50={acc50:.4f}, mIoU={miou:.4f}"
    )


if __name__ == "__main__":
    main()
