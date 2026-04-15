#!/usr/bin/env python3
"""ScanNet200 dataset helpers."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def _load_data():
    with open(_DATA_DIR / 'scannet200_classes.json') as f:
        return json.load(f)


def defaults() -> Dict[str, Optional[str]]:
    return {
        'data_root': 'data/scannet200',
        'out_root_gt': 'results/graphs/scannet200_gt',
        'out_root_oneformer': 'results/graphs/scannet200_oneformer',
        'results_dir_oneformer': 'results/oneformer3d_1xb4_scannet200'
    }


def load_class_names() -> List[str]:
    return _load_data()['class_names']


def load_sem_mapping() -> List[int]:
    return _load_data()['sem_mapping']


def raw_id_to_class(sem_mapping: List[int]) -> Dict[int, int]:
    return {int(raw): int(i) for i, raw in enumerate(sem_mapping)}


def swap_chair_floor_ids(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels
    labels = labels.copy()
    mask2 = labels == 2
    mask3 = labels == 3
    labels[mask2] = 3
    labels[mask3] = 2
    return labels


def swap_chair_floor_id(raw_id: int) -> int:
    if raw_id == 2:
        return 3
    if raw_id == 3:
        return 2
    return raw_id


def sem_id_to_label(raw_id: int, class_names: List[str], raw_id_to_class_map: Dict[int, int]) -> Optional[str]:
    raw_id = swap_chair_floor_id(raw_id)
    cls_idx = raw_id_to_class_map.get(int(raw_id))
    if cls_idx is None or cls_idx < 0 or cls_idx >= len(class_names):
        return None
    return class_names[cls_idx]


def map_oneformer_labels(inst_labels: np.ndarray, sem_mapping: np.ndarray) -> np.ndarray:
    inst_labels_raw = np.full(inst_labels.shape, -1, dtype=np.int64)
    train_ids = inst_labels + 2
    valid = (train_ids >= 0) & (train_ids < len(sem_mapping))
    inst_labels_raw[valid] = sem_mapping[train_ids[valid]]
    inst_labels_raw = swap_chair_floor_ids(inst_labels_raw)
    return inst_labels_raw
