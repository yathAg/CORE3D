#!/usr/bin/env python3
"""Filesystem IO helpers for graph building."""

import os.path as osp
from typing import Dict, List, Tuple

import numpy as np


def load_all_infos(data_root: str, infos_prefix: str):
    import pickle

    infos_all = []
    metainfo = None
    for split in ['train', 'val', 'test']:
        p = osp.join(data_root, f'{infos_prefix}_{split}.pkl')
        if not osp.exists(p):
            continue
        with open(p, 'rb') as f:
            data = pickle.load(f)
        infos_all.extend(data.get('data_list', []))
        if metainfo is None:
            metainfo = data.get('metainfo', {})
    return infos_all, metainfo or {}


def load_masks(data_root: str, instance_rel: str, semantic_rel: str) -> Tuple[np.ndarray, np.ndarray]:
    inst_path = osp.join(data_root, 'instance_mask', instance_rel)
    sem_path = osp.join(data_root, 'semantic_mask', semantic_rel)
    inst = np.fromfile(inst_path, dtype=np.int64)
    sem = np.fromfile(sem_path, dtype=np.int64)
    return inst, sem


def load_points(data_root: str, points_rel: str) -> np.ndarray:
    pts_path = osp.join(data_root, 'points', points_rel)
    pts = np.fromfile(pts_path, dtype=np.float32)
    if pts.size % 6 != 0:
        raise ValueError(f'Unexpected point feature length {pts.size} in {pts_path} (not divisible by 6)')
    return pts.reshape(-1, 6)
