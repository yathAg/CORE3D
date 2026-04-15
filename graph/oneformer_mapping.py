#!/usr/bin/env python3
"""OneFormer3D prediction to GT instance matching utilities."""
from typing import Dict

import numpy as np

# IoU threshold used to align predicted instances to GT ids.
PRED_MATCH_IOU_THR = 0.25


def _majority_label(labels: np.ndarray) -> int:
    if labels.size == 0:
        return -1
    if labels.min() < 0:
        labels = labels[labels >= 0]
        if labels.size == 0:
            return -1
    labels = labels.astype(np.int64, copy=False)
    counts = np.bincount(labels)
    max_count = counts.max()
    cand = np.flatnonzero(counts == max_count)
    if cand.size == 1:
        return int(cand[0])
    cand_set = set(cand.tolist())
    for val in labels:
        if val in cand_set:
            return int(val)
    return int(cand[0])


def _compute_gt_stats(gt_inst: np.ndarray, gt_sem: np.ndarray):
    max_gt_id = int(gt_inst.max()) if gt_inst.size else 0
    gt_sizes = np.bincount(gt_inst, minlength=max_gt_id + 1)
    gt_class = np.full(max_gt_id + 1, -1, dtype=int)
    if gt_inst.size == 0:
        return gt_sizes, gt_class
    order = np.argsort(gt_inst, kind='mergesort')
    gt_inst_sorted = gt_inst[order]
    gt_sem_sorted = gt_sem[order]
    uniq_gt, starts, counts = np.unique(gt_inst_sorted, return_index=True, return_counts=True)
    for gid, start, count in zip(uniq_gt, starts, counts):
        if gid <= 0:
            continue
        end = int(start + count)
        labels = gt_sem_sorted[start:end]
        gt_class[gid] = _majority_label(labels)
    return gt_sizes, gt_class


def match_preds_to_gt(inst_masks: np.ndarray,
                      inst_labels: np.ndarray,
                      inst_scores: np.ndarray,
                      gt_inst: np.ndarray,
                      gt_sem: np.ndarray,
                      iou_thr: float = PRED_MATCH_IOU_THR) -> Dict[int, Dict]:
    """Return best prediction per GT id (same class, IoU>=thr, highest score)."""
    gt_sizes, gt_class = _compute_gt_stats(gt_inst, gt_sem)
    max_gt_id = len(gt_class) - 1

    best: Dict[int, Dict] = {}

    for p_idx, pmask in enumerate(inst_masks):
        p_mask = pmask if pmask.dtype == np.bool_ else pmask.astype(bool)
        p_points = int(p_mask.sum())
        if p_points == 0:
            continue
        p_label_id = int(inst_labels[p_idx]) if p_idx < len(inst_labels) else -1
        if p_label_id < 0:
            continue

        gt_ids_in_pred = gt_inst[p_mask]
        if gt_ids_in_pred.size == 0:
            continue
        counts = np.bincount(gt_ids_in_pred, minlength=max_gt_id + 1)
        cand = np.nonzero(counts)[0]
        if cand.size == 0:
            continue
        cand = cand[cand != 0]
        if cand.size == 0:
            continue
        class_match = gt_class[cand] == p_label_id
        cand = cand[class_match]
        if cand.size == 0:
            continue
        inter = counts[cand].astype(np.float32, copy=False)
        union = (p_points + gt_sizes[cand] - inter).astype(np.float32, copy=False)
        ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        best_idx = int(np.argmax(ious))
        best_gid = int(cand[best_idx])
        best_iou = float(ious[best_idx])
        if best_iou < iou_thr:
            continue

        score = float(inst_scores[p_idx]) if p_idx < len(inst_scores) else 0.0
        prev = best.get(best_gid)
        if prev is None or score > prev['score']:
            best[best_gid] = {
                'pred_idx': p_idx,
                'mask': p_mask,
                'points': p_points,
                'label_id': int(gt_class[best_gid]),
                'score': score,
                'iou': best_iou
            }
    return best
