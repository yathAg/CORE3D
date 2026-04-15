#!/usr/bin/env python3
"""Compute mask IoU for Surprise3D predictions (JSONL) using ScanNet++ masks.

Reads preds_<split>.jsonl from a run directory, loads per-scene graphs to map
predicted ids -> OneFormer3D instance masks (if applicable), and computes IoU
in two modes:
  - multi: union of all predicted ids vs union of GT instance masks
  - single_best: best single-id IoU vs union of GT instance masks
Reports overall and per-question-type breakdowns for both modes and writes
separate CSV/JSON summaries.
"""

import argparse
import csv
import json
import os
import os.path as osp
import pickle
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


def _resolve_defaults() -> Dict[str, str]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    return {
        "repo_root": str(repo_root),
        "graph_root": str(repo_root / "results/graphs/scannetpp_oneformer/scenes"),
        "data_root": str(repo_root / "data/scannetpp"),
        "pred_root": str(repo_root / "results/oneformer3d_1xb4_scannetpp_spconv_sdpa_qthr05_25k"),
        "infos_root": str(repo_root / "data/scannetpp"),
    }


def _load_graph_run_args(graph_root: str) -> Dict:
    if not graph_root:
        return {}
    run_args_path = osp.join(osp.dirname(osp.normpath(graph_root)), "run_args.json")
    if not osp.exists(run_args_path):
        return {}
    try:
        with open(run_args_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


QTYPE_LABELS = {
    "cs": "Common-sense",
    "hi": "Human Intention",
    "first_view": "Narrative Perspective",
    "camera_view": "Parametric Perspective",
    "relative_position": "Relative Position",
    "abs": "Absolute Distance",
}
QTYPE_ORDER = ["cs", "hi", "first_view", "camera_view", "relative_position", "abs"]


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
                "question_type": rec.get("question_type") or "unknown",
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


def _load_graph_mapping(graph: Dict) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    mapping_raw = graph.get("pred_alignment") or {}
    for k, v in mapping_raw.items():
        if not isinstance(v, dict):
            continue
        pred_idx = v.get("pred_idx")
        if pred_idx is None:
            continue
        try:
            gid = int(k)
            mapping[gid] = int(pred_idx)
        except Exception:
            continue
    return mapping


def _load_gt_inst(data_root: str, scene_id: str) -> Optional[np.ndarray]:
    gt_path = osp.join(data_root, "instance_mask", f"{scene_id}.bin")
    if not osp.exists(gt_path):
        return None
    return np.fromfile(gt_path, dtype=np.int64)


def _load_pred_masks(pred_root: str, scene_id: str) -> Optional[np.ndarray]:
    pred_path = osp.join(pred_root, f"{scene_id}.pth")
    if not osp.exists(pred_path):
        return None
    pred_obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    inst_masks = np.asarray(pred_obj.get("instance_masks"))
    if inst_masks.size == 0 or inst_masks.ndim != 2:
        return None
    return inst_masks


def _load_pred_sample_indices(pred_root: str, scene_id: str) -> Optional[np.ndarray]:
    pred_path = osp.join(pred_root, f"{scene_id}.pth")
    if not osp.exists(pred_path):
        return None
    pred_obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    si = pred_obj.get("sample_indices")
    if si is None:
        return None
    return np.asarray(si, dtype=np.int64)


def _compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    return (inter / union) if union > 0 else 0.0


def _get_num_points(points_path: str) -> int:
    size = os.path.getsize(points_path)
    if size % (6 * 4) != 0:
        raise ValueError(f"Unexpected byte size for points file: {points_path}")
    return size // (6 * 4)


def _load_infos_by_split(infos_root: str) -> Dict[str, List[Dict]]:
    infos_by_split: Dict[str, List[Dict]] = {}
    for split in ["train", "val", "test"]:
        p = osp.join(infos_root, f"scannetpp_oneformer3d_infos_{split}.pkl")
        if not osp.exists(p):
            continue
        with open(p, "rb") as f:
            data = pickle.load(f)
        infos_by_split[split] = data.get("data_list", [])
    return infos_by_split


def _build_sampling_choices(
    scene_ids: set,
    infos_by_split: Dict[str, List[Dict]],
    data_root: str,
    num_points: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Reproduce PointSample_ choices with RNG reset per split."""
    choices_map: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        infos = infos_by_split.get(split, [])
        if not infos:
            continue
        rng = np.random.RandomState(seed)
        for info in infos:
            inst_rel = info.get("pts_instance_mask_path")
            points_rel = info.get("lidar_points", {}).get("lidar_path")
            if not inst_rel or not points_rel:
                continue
            scene_id = Path(inst_rel).stem
            points_path = osp.join(data_root, "points", points_rel)
            if not osp.exists(points_path):
                raise FileNotFoundError(f"Missing points file: {points_path}")
            n_points = _get_num_points(points_path)
            num_samples = min(num_points, n_points)
            choices = rng.choice(n_points, num_samples, replace=True)
            if scene_id in scene_ids:
                choices_map[scene_id] = choices
    return choices_map


def _init_stats() -> Dict[str, float]:
    return {
        "evaluated": 0,
        "skipped": 0,
        "iou_sum": 0.0,
        "iou25_true": 0,
        "iou50_true": 0,
        "parse_fail": 0,
        "missing_pred_map": 0,
        "missing_pred_idx": 0,
        "missing_files": 0,
    }


def _update_stats(
    stat: Dict[str, float],
    *,
    iou: Optional[float] = None,
    parse_fail: bool = False,
    missing_pred_map: bool = False,
    missing_pred_idx: bool = False,
    missing_files: bool = False,
    skipped: bool = False,
):
    if skipped:
        stat["skipped"] += 1
        if missing_files:
            stat["missing_files"] += 1
        return
    stat["evaluated"] += 1
    if iou is not None:
        stat["iou_sum"] += iou
        if iou >= 0.25:
            stat["iou25_true"] += 1
        if iou >= 0.5:
            stat["iou50_true"] += 1
    if parse_fail:
        stat["parse_fail"] += 1
    if missing_pred_map:
        stat["missing_pred_map"] += 1
    if missing_pred_idx:
        stat["missing_pred_idx"] += 1
    if missing_files:
        stat["missing_files"] += 1


def _finalize_stats(stat: Dict[str, float]) -> Dict[str, float]:
    evaluated = int(stat["evaluated"])
    mean_iou = (stat["iou_sum"] / evaluated) if evaluated else None
    iou25 = (stat["iou25_true"] / evaluated) if evaluated else None
    iou50 = (stat["iou50_true"] / evaluated) if evaluated else None
    return {
        "evaluated": evaluated,
        "skipped": int(stat["skipped"]),
        "mean_iou": mean_iou,
        "iou25": iou25,
        "iou50": iou50,
        "parse_fail": int(stat["parse_fail"]),
        "missing_pred_map": int(stat["missing_pred_map"]),
        "missing_pred_idx": int(stat["missing_pred_idx"]),
        "missing_files": int(stat["missing_files"]),
    }


def main():
    defaults = _resolve_defaults()
    ap = argparse.ArgumentParser(description="Mask IoU evaluation for Surprise3D predictions.")
    ap.add_argument("--run-dir", required=True, help="Run folder containing preds_<split>.jsonl")
    ap.add_argument("--split", required=True, help="Split name (train/val/all)")
    ap.add_argument("--graph-root", default=None, help="Override graph root for scene JSONs")
    ap.add_argument("--data-root", default=None, help="Override ScanNet++ data root")
    ap.add_argument("--pred-root", default=None, help="Override OneFormer3D prediction root")
    ap.add_argument("--infos-root", default=None, help="Override location of scannetpp_oneformer3d_infos_*.pkl")
    ap.add_argument("--numpy-seed", type=int, default=None, help="Override numpy seed for point sampling")
    ap.add_argument("--num-points", type=int, default=None, help="PointSample_ num_points used during inference")
    ap.add_argument("--no-json", action="store_true", help="Skip writing metrics_iou_*.json")
    args = ap.parse_args()

    run_dir = args.run_dir
    split = args.split
    write_json = not args.no_json
    preds_path = osp.join(run_dir, f"preds_{split}.jsonl")
    if not osp.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    run_params = _find_run_params(run_dir, split)
    graph_root = args.graph_root or (
        run_params.get("graph_root_resolved") if run_params else None
    ) or defaults["graph_root"]

    graph_run_args = _load_graph_run_args(graph_root)
    data_root = args.data_root or graph_run_args.get("data_root") or defaults["data_root"]
    pred_root = args.pred_root or graph_run_args.get("results_dir") or defaults["pred_root"]
    infos_root = args.infos_root or data_root or defaults["infos_root"]
    num_points = args.num_points or graph_run_args.get("num_points") or 20000
    numpy_seed = args.numpy_seed or graph_run_args.get("numpy_seed_used")
    object_id_shift = graph_run_args.get("object_id_shift", 0)

    entries_by_scene, total_entries, parse_fails = _load_predictions(preds_path)
    if not entries_by_scene:
        print("No entries found in predictions file.")
        return

    scene_ids = set(entries_by_scene.keys())
    graphs: Dict[str, Optional[Dict]] = {}
    needs_sampling = False
    for scene_id in scene_ids:
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        graph = None
        if osp.exists(graph_path):
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
            except Exception:
                graph = None
        graphs[scene_id] = graph
        if graph and not _is_gt_graph(graph):
            needs_sampling = True

    sampling_choices: Dict[str, np.ndarray] = {}
    if needs_sampling:
        # Prefer sample_indices saved in the pred .pth files (current graph pipeline).
        missing_si = []
        for scene_id in scene_ids:
            si = _load_pred_sample_indices(pred_root, scene_id)
            if si is not None:
                sampling_choices[scene_id] = si
            else:
                missing_si.append(scene_id)
        if missing_si:
            # Fallback: replay PointSample_ RNG for scenes without sample_indices.
            if numpy_seed is None:
                raise ValueError(
                    "Missing numpy seed for PointSample_. Pass --numpy-seed or ensure run_args.json "
                    f"contains numpy_seed_used. Scenes without sample_indices in pred files: {len(missing_si)}"
                )
            infos_by_split = _load_infos_by_split(infos_root)
            replayed = _build_sampling_choices(
                set(missing_si), infos_by_split, data_root, num_points, numpy_seed
            )
            sampling_choices.update(replayed)

    overall = _init_stats()
    stats_by_qtype: Dict[str, Dict[str, float]] = {}
    scene_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def ensure_qtype(stats_by_qtype: Dict[str, Dict[str, float]], qtype: str) -> Dict[str, float]:
        if qtype not in stats_by_qtype:
            stats_by_qtype[qtype] = _init_stats()
        return stats_by_qtype[qtype]

    def ensure_scene_qtype(scene_id: str, qtype: str) -> Dict[str, float]:
        qdict = scene_stats.setdefault(scene_id, {})
        if qtype not in qdict:
            qdict[qtype] = _init_stats()
        return qdict[qtype]

    for scene_id, entries in entries_by_scene.items():
        graph = graphs.get(scene_id)
        if not graph:
            for e in entries:
                qtype = e["question_type"]
                _update_stats(overall, skipped=True, missing_files=True)
                _update_stats(ensure_qtype(stats_by_qtype, qtype), skipped=True, missing_files=True)
                _update_stats(ensure_scene_qtype(scene_id, qtype), skipped=True, missing_files=True)
            continue

        use_gt_masks = _is_gt_graph(graph)
        mapping = _load_graph_mapping(graph) if not use_gt_masks else {}

        if use_gt_masks:
            gt_inst = _load_gt_inst(data_root, scene_id)
            inst_masks = None
            if gt_inst is None:
                for e in entries:
                    qtype = e["question_type"]
                    _update_stats(overall, skipped=True, missing_files=True)
                    _update_stats(ensure_qtype(stats_by_qtype, qtype), skipped=True, missing_files=True)
                    _update_stats(ensure_scene_qtype(scene_id, qtype), skipped=True, missing_files=True)
                continue
        else:
            gt_full = _load_gt_inst(data_root, scene_id)
            inst_masks = _load_pred_masks(pred_root, scene_id)
            choices = sampling_choices.get(scene_id)
            if gt_full is None or inst_masks is None or choices is None:
                for e in entries:
                    qtype = e["question_type"]
                    _update_stats(overall, skipped=True, missing_files=True)
                    _update_stats(ensure_qtype(stats_by_qtype, qtype), skipped=True, missing_files=True)
                    _update_stats(ensure_scene_qtype(scene_id, qtype), skipped=True, missing_files=True)
                continue
            n = min(len(choices), inst_masks.shape[1])
            if n <= 0:
                for e in entries:
                    qtype = e["question_type"]
                    _update_stats(overall, skipped=True, missing_files=True)
                    _update_stats(ensure_qtype(stats_by_qtype, qtype), skipped=True, missing_files=True)
                    _update_stats(ensure_scene_qtype(scene_id, qtype), skipped=True, missing_files=True)
                continue
            choices = choices[:n]
            gt_inst = gt_full[choices]
            inst_masks = inst_masks[:, :n].astype(bool)

        for e in entries:
            qtype = e["question_type"]
            stat_overall = overall
            stat_qtype = ensure_qtype(stats_by_qtype, qtype)
            stat_scene = ensure_scene_qtype(scene_id, qtype)

            if e.get("skipped"):
                _update_stats(stat_overall, skipped=True)
                _update_stats(stat_qtype, skipped=True)
                _update_stats(stat_scene, skipped=True)
                continue

            gt_ids = e.get("gt_ids", [])
            pred_ids = e.get("pred_ids", [])
            parse_fail = bool(e.get("parse_fail", False))
            if parse_fail:
                pred_ids = []
            if use_gt_masks and object_id_shift:
                gt_ids = [i - object_id_shift for i in gt_ids]
                pred_ids = [i - object_id_shift for i in pred_ids]

            missing_pred_map = False
            missing_pred_idx = False

            if use_gt_masks:
                gt_mask = np.isin(gt_inst, gt_ids)
                best_iou = 0.0
                for pid in pred_ids:
                    pred_mask_single = gt_inst == pid
                    best_iou = max(best_iou, _compute_iou(gt_mask, pred_mask_single))
            else:
                pred_idxs = []
                for pid in pred_ids:
                    pred_idx = mapping.get(pid)
                    if pred_idx is None:
                        missing_pred_map = True
                        continue
                    if pred_idx < 0 or pred_idx >= inst_masks.shape[0]:
                        missing_pred_idx = True
                        continue
                    pred_idxs.append(pred_idx)
                gt_mask = np.isin(gt_inst, gt_ids)
                best_iou = 0.0
                for pred_idx in pred_idxs:
                    pred_mask_single = inst_masks[pred_idx]
                    best_iou = max(best_iou, _compute_iou(gt_mask, pred_mask_single))

            _update_stats(
                stat_overall,
                iou=best_iou,
                parse_fail=parse_fail,
                missing_pred_map=missing_pred_map,
                missing_pred_idx=missing_pred_idx,
            )
            _update_stats(
                stat_qtype,
                iou=best_iou,
                parse_fail=parse_fail,
                missing_pred_map=missing_pred_map,
                missing_pred_idx=missing_pred_idx,
            )
            _update_stats(
                stat_scene,
                iou=best_iou,
                parse_fail=parse_fail,
                missing_pred_map=missing_pred_map,
                missing_pred_idx=missing_pred_idx,
            )

    metrics = {
        "overall": _finalize_stats(overall),
        "per_question_type": {
            qt: _finalize_stats(s) for qt, s in sorted(stats_by_qtype.items())
        },
        "total_queries": total_entries,
        "parse_fails": parse_fails,
    }

    if write_json:
        metrics_json_path = osp.join(run_dir, f"metrics_iou_{split}.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _write_csv(path: str, overall_stat: Dict[str, float], stats_by_qtype: Dict[str, Dict[str, float]]):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question_type",
                    "evaluated",
                    "skipped",
                    "mIoU",
                    "Acc@0.25",
                    "Acc@0.50",
                    "parse_fail",
                    "missing_pred_map",
                    "missing_pred_idx",
                    "missing_files",
                ]
            )
            ordered_qtypes = [q for q in QTYPE_ORDER if q in stats_by_qtype]
            other_qtypes = sorted(q for q in stats_by_qtype if q not in QTYPE_ORDER)
            for qt in ordered_qtypes + other_qtypes:
                fmt = _finalize_stats(stats_by_qtype[qt])
                display_qt = QTYPE_LABELS.get(qt, qt)
                writer.writerow(
                    [
                        display_qt,
                        fmt["evaluated"],
                        fmt["skipped"],
                        f"{fmt['mean_iou']:.4f}" if fmt["mean_iou"] is not None else "",
                        f"{fmt['iou25']:.4f}" if fmt["iou25"] is not None else "",
                        f"{fmt['iou50']:.4f}" if fmt["iou50"] is not None else "",
                        fmt["parse_fail"],
                        fmt["missing_pred_map"],
                        fmt["missing_pred_idx"],
                        fmt["missing_files"],
                    ]
                )
            overall_fmt = _finalize_stats(overall_stat)
            writer.writerow(
                [
                    "overall",
                    overall_fmt["evaluated"],
                    overall_fmt["skipped"],
                    f"{overall_fmt['mean_iou']:.4f}" if overall_fmt["mean_iou"] is not None else "",
                    f"{overall_fmt['iou25']:.4f}" if overall_fmt["iou25"] is not None else "",
                    f"{overall_fmt['iou50']:.4f}" if overall_fmt["iou50"] is not None else "",
                    overall_fmt["parse_fail"],
                    overall_fmt["missing_pred_map"],
                    overall_fmt["missing_pred_idx"],
                    overall_fmt["missing_files"],
                ]
            )

    csv_path = osp.join(run_dir, f"metric_iou_{split}.csv")
    _write_csv(csv_path, overall, stats_by_qtype)

    overall_mean = metrics["overall"]["mean_iou"]
    overall_str = f"{overall_mean:.4f}" if overall_mean is not None else "n/a"
    print(f"Saved IoU summary to {csv_path}. Overall mIoU={overall_str}")


if __name__ == "__main__":
    main()
