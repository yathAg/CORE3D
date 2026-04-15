#!/usr/bin/env python3
"""ScanNet++ dataset helpers (allowlist-driven semantics)."""

import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_ALLOWLIST = 'data/scannetpp/metadata/semantic_benchmark/top200_semantic.txt'
ALLOWLIST_FILENAME = 'top200_semantic.txt'

# Number of "stuff" classes prepended to top200_semantic before the
# 194 instance classes (top200_instance.txt). The OneFormer3D head
# outputs instance-class indices in [0, 194); add (STUFF_OFFSET + 1)
# to reach raw 1-based semantic ids in [7, 200].
STUFF_OFFSET = 6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def defaults() -> Dict[str, Optional[str]]:
    return {
        'data_root': 'data/scannetpp',
        'out_root_gt': 'results/graphs/scannetpp_gt',
        'out_root_oneformer': 'results/graphs/scannetpp_oneformer',
        'results_dir_oneformer': 'results/oneformer3d_spp/preds',
        'label_allowlist': DEFAULT_ALLOWLIST
    }


def map_oneformer_labels(inst_labels: np.ndarray) -> np.ndarray:
    raw = inst_labels.astype(np.int64) + (STUFF_OFFSET + 1)
    raw[inst_labels < 0] = -1
    return raw


def resolve_allowlist_path(data_root: str, label_allowlist: Optional[str]) -> Optional[str]:
    if label_allowlist and osp.exists(label_allowlist):
        return label_allowlist

    meta_root = osp.join(data_root, 'metadata', 'semantic_benchmark')
    candidate = osp.join(meta_root, ALLOWLIST_FILENAME)
    if osp.exists(candidate):
        return candidate

    fallback = _repo_root() / 'data' / 'scannetpp' / 'metadata' / 'semantic_benchmark' / ALLOWLIST_FILENAME
    if fallback.exists():
        return str(fallback)
    return None


def load_allowlist(path: str) -> List[str]:
    if not path or not osp.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def raw_id_to_class(class_names: List[str]) -> Dict[int, int]:
    return {i + 1: i for i in range(len(class_names))}


def sem_id_to_label(raw_id: int, class_names: List[str], raw_id_to_class_map: Dict[int, int]) -> Optional[str]:
    cls_idx = raw_id_to_class_map.get(int(raw_id))
    if cls_idx is None or cls_idx < 0 or cls_idx >= len(class_names):
        return None
    return class_names[cls_idx]


def init_stats() -> Dict[str, int]:
    return {'max_sem_id': 0, 'skipped_sem_id': 0}


def update_stats(stats: Dict[str, int], sem_id: int, used: bool) -> None:
    stats['max_sem_id'] = max(stats.get('max_sem_id', 0), int(sem_id))
    if not used:
        stats['skipped_sem_id'] = stats.get('skipped_sem_id', 0) + 1


def finalize_run_cfg(run_cfg: Dict[str, object],
                     stats: Dict[str, int],
                     allowlist_path: str,
                     num_classes: int) -> None:
    run_cfg['label_allowlist_resolved'] = allowlist_path
    run_cfg['num_classes'] = num_classes
    run_cfg['max_sem_id_seen'] = stats.get('max_sem_id', 0)
    run_cfg['skipped_instances_missing_label'] = stats.get('skipped_sem_id', 0)
