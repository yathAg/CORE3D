#!/usr/bin/env python3
"""Compute mask IoU accuracy for ScanRefer predictions (JSONL) using OneFormer3D masks.

Reads preds_<split>.jsonl from a run directory, loads per-scene graphs to map
GT ids -> pred instance indices, and computes IoU between the predicted mask
and the GT mask for each query. Outputs metric_iou_<split>.csv with totals only.
"""

import argparse
import csv
import json
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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


def _load_predictions(path: str) -> Tuple[Dict[str, List[Dict]], int, int]:
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
            gt_id_0_raw = rec.get("gt_object_id")
            gt_id_1_raw = rec.get("gt_object_id_1indexed")
            try:
                gt_id_0 = int(gt_id_0_raw) if gt_id_0_raw is not None else None
            except Exception:
                gt_id_0 = None
            try:
                gt_id_1 = int(gt_id_1_raw) if gt_id_1_raw is not None else None
            except Exception:
                gt_id_1 = None
            pred_raw = rec.get("prediction")
            parse_fail = bool(rec.get("parse_fail", False))
            try:
                pred_id = int(pred_raw) if pred_raw is not None else None
            except Exception:
                pred_id = None
            entry = {
                "scene_id": scene_id,
                "gt_id_0": gt_id_0,
                "gt_id_1": gt_id_1,
                "pred_id": pred_id,
                "parse_fail": parse_fail,
            }
            by_scene[scene_id].append(entry)
            total += 1
            if parse_fail:
                parse_fails += 1
    return by_scene, total, parse_fails


def _load_unique_multiple(path: str) -> Optional[Dict]:
    if not path or not osp.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _normalize_unique_multiple(lookup: Optional[Dict]) -> Optional[Dict[str, Dict[str, int]]]:
    if not lookup or not isinstance(lookup, dict):
        return None
    normalized: Dict[str, Dict[str, int]] = {}
    for scene_id, obj_map in lookup.items():
        if not isinstance(obj_map, dict):
            continue
        scene_norm: Dict[str, int] = {}
        for obj_id, val in obj_map.items():
            if isinstance(val, dict):
                vals = []
                for v in val.values():
                    try:
                        vals.append(int(v))
                    except Exception:
                        continue
                if not vals:
                    continue
                # Use majority vote (should be consistent across ann_ids).
                scene_norm[str(obj_id)] = 1 if sum(vals) >= (len(vals) / 2) else 0
            else:
                try:
                    scene_norm[str(obj_id)] = int(val)
                except Exception:
                    continue
        if scene_norm:
            normalized[str(scene_id)] = scene_norm
    return normalized or None


def _resolve_scanrefer_paths(repo_root: Path) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        repo_root / "data/Scanrefer/ScanRefer_filtered.json",
        repo_root / "data/ScanRefer/ScanRefer_filtered.json",
        repo_root / "data/scanrefer/ScanRefer_filtered.json",
    ]
    scanrefer_path = next((str(p) for p in candidates if p.exists()), None)
    labels_path = repo_root / "data/scannet200/meta_data/scannetv2-labels.combined.tsv"
    return scanrefer_path, str(labels_path) if labels_path.exists() else None


def _build_unique_multiple_lookup(
    scanrefer_path: str,
    labels_tsv_path: str,
    out_path: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    if not scanrefer_path or not labels_tsv_path:
        return None
    try:
        with open(scanrefer_path) as f:
            scanrefer_data = json.load(f)
    except Exception:
        return None

    # Build raw2label mapping from ScanNet TSV.
    type2class = {
        "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5,
        "window": 6, "bookshelf": 7, "picture": 8, "counter": 9, "desk": 10,
        "curtain": 11, "refrigerator": 12, "shower curtain": 13, "toilet": 14,
        "sink": 15, "bathtub": 16, "others": 17,
    }
    scannet_labels = set(type2class.keys())
    scannet2label = {label: i for i, label in enumerate(type2class.keys())}
    raw2label = {}
    try:
        import csv

        with open(labels_tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                if len(row) < 8:
                    continue
                raw_name = row[1]
                nyu40_name = row[7]
                if nyu40_name not in scannet_labels:
                    raw2label[raw_name] = scannet2label["others"]
                else:
                    raw2label[raw_name] = scannet2label[nyu40_name]
    except Exception:
        return None

    all_sem_labels: Dict[str, List[int]] = defaultdict(list)
    cache: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for data in scanrefer_data:
        scene_id = data.get("scene_id")
        object_id = data.get("object_id")
        object_name = data.get("object_name")
        if scene_id is None or object_id is None or object_name is None:
            continue
        if object_id not in cache[scene_id]:
            cache[scene_id][object_id] = True
            obj_name = " ".join(object_name.split("_"))
            sem_label = raw2label.get(obj_name, 17)
            all_sem_labels[scene_id].append(sem_label)

    all_sem_labels = {sid: np.array(vals) for sid, vals in all_sem_labels.items()}

    unique_multiple_lookup: Dict[str, Dict[str, int]] = defaultdict(dict)
    for data in scanrefer_data:
        scene_id = data.get("scene_id")
        object_id = data.get("object_id")
        object_name = data.get("object_name")
        if scene_id is None or object_id is None or object_name is None:
            continue
        obj_name = " ".join(object_name.split("_"))
        sem_label = raw2label.get(obj_name, 17)
        unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
        unique_multiple_lookup[str(scene_id)][str(object_id)] = int(unique_multiple)

    if out_path:
        try:
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(unique_multiple_lookup, f, indent=2)
        except Exception:
            pass

    return unique_multiple_lookup


def _resolve_defaults() -> Dict[str, str]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    scanrefer_path, labels_path = _resolve_scanrefer_paths(repo_root)
    unique_lookup_default = None
    for candidate in (
        repo_root / "data/Scanrefer/scanrefer_unique_multiple_lookup.json",
        repo_root / "data/ScanRefer/scanrefer_unique_multiple_lookup.json",
        repo_root / "3DGraphLLM/annotations/scanrefer_unique_multiple_lookup.json",
    ):
        if candidate.exists():
            unique_lookup_default = str(candidate)
            break
    if unique_lookup_default is None:
        unique_lookup_default = str(repo_root / "data/Scanrefer/scanrefer_unique_multiple_lookup.json")
    return {
        "oneformer_root": str(repo_root),
        "graph_root": str(repo_root / "results/graphs/scannet200_oneformer/scenes"),
        "data_root": str(repo_root / "data/scannet200"),
        "pred_root": str(repo_root / "results/oneformer3d_1xb4_scannet200"),
        "unique_lookup": unique_lookup_default,
        "scanrefer_path": scanrefer_path or "",
        "scannet_labels_path": labels_path or "",
    }


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


def _load_gt_inst(data_root: str, scene_id: str) -> Optional[np.ndarray]:
    gt_path = osp.join(data_root, "instance_mask", f"{scene_id}.bin")
    if not osp.exists(gt_path):
        return None
    return np.fromfile(gt_path, dtype=np.int64)


def _load_pred_masks(data_root: str, pred_root: str, scene_id: str):
    gt_inst = _load_gt_inst(data_root, scene_id)
    pred_path = osp.join(pred_root, f"{scene_id}.pth")
    if gt_inst is None or not osp.exists(pred_path):
        return None, None
    pred_obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    inst_masks = np.asarray(pred_obj.get("instance_masks"))
    if inst_masks.size == 0:
        return None, None
    if inst_masks.ndim != 2:
        return None, None
    n = min(len(gt_inst), inst_masks.shape[1])
    if n == 0:
        return None, None
    return gt_inst[:n], inst_masks[:, :n]


def _is_gt_graph(graph: Optional[Dict]) -> bool:
    if not graph:
        return False
    if graph.get("id_source") == "gt" or graph.get("prediction_source") == "gt":
        return True
    if graph.get("prediction_source") is not None:
        return False
    pred_alignment = graph.get("pred_alignment")
    return pred_alignment is None


def _compute_scene_iou_stats(
    entries: List[Dict],
    mapping: Dict[int, int],
    gt_inst: Optional[np.ndarray],
    inst_masks: Optional[np.ndarray],
    unique_lookup: Optional[Dict[str, Dict[str, int]]],
    scene_id: str,
    thresholds: Tuple[float, float],
    use_gt_masks: bool,
):
    t25, t50 = thresholds
    stats = {
        "total": 0,
        "correct25": 0,
        "correct50": 0,
        "unique_total": 0,
        "unique_correct25": 0,
        "unique_correct50": 0,
        "multiple_total": 0,
        "multiple_correct25": 0,
        "multiple_correct50": 0,
        "unique_iou_sum": 0.0,
        "multiple_iou_sum": 0.0,
        "parse_fail": 0,
        "missing_pred_map": 0,
        "missing_pred_idx": 0,
        "missing_files": 0,
        "iou_sum": 0.0,
    }

    if gt_inst is None or (not use_gt_masks and inst_masks is None):
        for e in entries:
            stats["total"] += 1
            if e.get("parse_fail"):
                stats["parse_fail"] += 1
            if unique_lookup is not None:
                gt_id_0 = e.get("gt_id_0")
                subset = unique_lookup.get(scene_id, {}).get(str(gt_id_0))
                if subset == 0:
                    stats["unique_total"] += 1
                elif subset == 1:
                    stats["multiple_total"] += 1
        stats["missing_files"] += len(entries)
        return stats

    gt_max = int(gt_inst.max()) if gt_inst.size else 0
    gt_sizes = np.bincount(gt_inst, minlength=gt_max + 1)

    pred_cache: Dict[int, Tuple[int, np.ndarray]] = {}
    if not use_gt_masks:
        needed_pred_idxs = set()
        for e in entries:
            if e.get("parse_fail"):
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
                counts = np.zeros_like(gt_sizes)
            else:
                counts = np.bincount(gt_inst[p_mask], minlength=len(gt_sizes))
            pred_cache[pred_idx] = (p_size, counts)

    for e in entries:
        stats["total"] += 1
        parse_fail = bool(e.get("parse_fail"))
        if parse_fail:
            stats["parse_fail"] += 1
        pred_id = e.get("pred_id")
        gt_id = e.get("gt_id_1")

        subset = None
        if unique_lookup is not None:
            try:
                gt_id_0 = e.get("gt_id_0")
                subset = unique_lookup.get(scene_id, {}).get(str(gt_id_0))
            except Exception:
                subset = None

        if subset == 0:
            stats["unique_total"] += 1
        elif subset == 1:
            stats["multiple_total"] += 1

        if parse_fail or pred_id is None or gt_id is None:
            continue

        if use_gt_masks:
            if pred_id is None or pred_id < 0 or pred_id >= len(gt_sizes):
                stats["missing_pred_map"] += 1
                continue
            p_size = int(gt_sizes[pred_id])
            counts = None
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
        if gt_id >= len(gt_sizes) or gt_id < 0:
            continue
        if use_gt_masks:
            inter = int(gt_sizes[gt_id]) if pred_id == gt_id else 0
            union = p_size + int(gt_sizes[gt_id]) - inter
        else:
            inter = int(counts[gt_id])
            union = p_size + int(gt_sizes[gt_id]) - inter
        iou = (inter / union) if union > 0 else 0.0
        stats["iou_sum"] += float(iou)
        if subset == 0:
            stats["unique_iou_sum"] += float(iou)
        elif subset == 1:
            stats["multiple_iou_sum"] += float(iou)

        if iou >= t25:
            stats["correct25"] += 1
            if subset == 0:
                stats["unique_correct25"] += 1
            elif subset == 1:
                stats["multiple_correct25"] += 1
        if iou >= t50:
            stats["correct50"] += 1
            if subset == 0:
                stats["unique_correct50"] += 1
            elif subset == 1:
                stats["multiple_correct50"] += 1

    return stats


def main():
    defaults = _resolve_defaults()
    ap = argparse.ArgumentParser(description="Mask IoU evaluation for ScanRefer JSONL predictions.")
    ap.add_argument("--run-dir", required=True, help="Run folder containing preds_<split>.jsonl")
    ap.add_argument("--split", required=True, help="Split name (train/val/all)")
    ap.add_argument("--graph-root", default=None, help="Override graph root for scene JSONs")
    ap.add_argument("--data-root", default=None, help="Override ScanNet200 data root")
    ap.add_argument("--pred-root", default=None, help="Override OneFormer3D prediction root")
    ap.add_argument("--unique-lookup", default=None, help="Override unique/multiple lookup JSON")
    ap.add_argument("--scanrefer-path", default=None, help="Path to ScanRefer_filtered.json (for building lookup)")
    ap.add_argument("--scannet-labels-path", default=None, help="Path to scannetv2-labels.combined.tsv (for building lookup)")
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
    unique_lookup_path = args.unique_lookup or defaults["unique_lookup"]
    scanrefer_path = args.scanrefer_path or defaults.get("scanrefer_path")
    scannet_labels_path = args.scannet_labels_path or defaults.get("scannet_labels_path")

    unique_lookup = _load_unique_multiple(unique_lookup_path)
    if unique_lookup is None:
        unique_lookup = _build_unique_multiple_lookup(
            scanrefer_path=scanrefer_path,
            labels_tsv_path=scannet_labels_path,
            out_path=unique_lookup_path,
        )
    unique_lookup = _normalize_unique_multiple(unique_lookup)
    entries_by_scene, total_entries, parse_fails = _load_predictions(preds_path)

    totals = {
        "total": 0,
        "correct25": 0,
        "correct50": 0,
        "unique_total": 0,
        "unique_correct25": 0,
        "unique_correct50": 0,
        "multiple_total": 0,
        "multiple_correct25": 0,
        "multiple_correct50": 0,
        "unique_iou_sum": 0.0,
        "multiple_iou_sum": 0.0,
        "parse_fail": 0,
        "missing_pred_map": 0,
        "missing_pred_idx": 0,
        "missing_files": 0,
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
        if use_gt_masks:
            gt_inst = _load_gt_inst(data_root, scene_id)
            inst_masks = None
        else:
            gt_inst, inst_masks = _load_pred_masks(data_root, pred_root, scene_id)
        stats = _compute_scene_iou_stats(
            entries,
            mapping,
            gt_inst,
            inst_masks,
            unique_lookup,
            scene_id,
            thresholds,
            use_gt_masks,
        )
        for k in totals:
            totals[k] += stats.get(k, 0)

    total_eval = totals["total"]
    acc25 = (totals["correct25"] / total_eval) if total_eval else 0.0
    acc50 = (totals["correct50"] / total_eval) if total_eval else 0.0
    miou = (totals["iou_sum"] / total_eval) if total_eval else 0.0
    uniq_eval = totals["unique_total"]
    mult_eval = totals["multiple_total"]
    uniq_acc25 = (totals["unique_correct25"] / uniq_eval) if uniq_eval else 0.0
    uniq_acc50 = (totals["unique_correct50"] / uniq_eval) if uniq_eval else 0.0
    mult_acc25 = (totals["multiple_correct25"] / mult_eval) if mult_eval else 0.0
    mult_acc50 = (totals["multiple_correct50"] / mult_eval) if mult_eval else 0.0
    uniq_miou = (totals["unique_iou_sum"] / uniq_eval) if uniq_eval else 0.0
    mult_miou = (totals["multiple_iou_sum"] / mult_eval) if mult_eval else 0.0

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
                "unique_evaluated",
                "Unique Correct@0.25",
                "Unique Acc@0.25",
                "Unique Correct@0.50",
                "Unique Acc@0.50",
                "Unique mIoU",
                "multiple_evaluated",
                "Multiple Correct@0.25",
                "Multiple Acc@0.25",
                "Multiple Correct@0.50",
                "Multiple Acc@0.50",
                "Multiple mIoU",
                "parse_fail",
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
                uniq_eval,
                totals["unique_correct25"],
                f"{uniq_acc25:.4f}" if uniq_eval else "",
                totals["unique_correct50"],
                f"{uniq_acc50:.4f}" if uniq_eval else "",
                f"{uniq_miou:.4f}" if uniq_eval else "",
                mult_eval,
                totals["multiple_correct25"],
                f"{mult_acc25:.4f}" if mult_eval else "",
                totals["multiple_correct50"],
                f"{mult_acc50:.4f}" if mult_eval else "",
                f"{mult_miou:.4f}" if mult_eval else "",
                totals["parse_fail"],
                totals["missing_pred_map"],
                totals["missing_pred_idx"],
                totals["missing_files"],
            ]
        )

    print(
        f"Saved IoU summary to {out_csv}. "
        f"Acc@0.25={acc25:.4f}, Acc@0.50={acc50:.4f}, "
        f"Unique Acc@0.25={uniq_acc25:.4f}, Multiple Acc@0.25={mult_acc25:.4f}"
    )


if __name__ == "__main__":
    main()
