#!/usr/bin/env python3
"""Dataset registry for graph builders."""

from dataclasses import dataclass
import os.path as osp
from typing import Callable, Dict, List, Optional

import numpy as np

from . import scannet200
from . import scannetpp


@dataclass
class DatasetSpec:
    key: str
    class_names: List[str]
    raw_id_to_class: Dict[int, int]
    defaults: Dict[str, Optional[str]]
    infos_prefix: str
    sem_id_to_label: Callable[[int], Optional[str]]
    map_oneformer_labels: Optional[Callable[[np.ndarray], np.ndarray]] = None
    stats: Optional[Dict[str, int]] = None
    update_stats: Optional[Callable[[int, bool], None]] = None
    finalize_run_cfg: Optional[Callable[[Dict[str, object]], None]] = None
    allowlist_path: Optional[str] = None
    keep_background_default: bool = False


def load_dataset(dataset_key: str, data_root: Optional[str], label_allowlist: Optional[str]) -> DatasetSpec:
    if dataset_key == 's200':
        defaults = scannet200.defaults()
        if data_root is None:
            data_root = defaults['data_root']
        class_names = scannet200.load_class_names()
        sem_mapping = np.array(scannet200.load_sem_mapping(), dtype=np.int64)
        raw_id_to_class_map = scannet200.raw_id_to_class(sem_mapping.tolist())

        def sem_id_to_label(raw_id: int) -> Optional[str]:
            return scannet200.sem_id_to_label(raw_id, class_names, raw_id_to_class_map)

        def map_oneformer_labels(inst_labels: np.ndarray) -> np.ndarray:
            return scannet200.map_oneformer_labels(inst_labels, sem_mapping)

        return DatasetSpec(
            key='s200',
            class_names=class_names,
            raw_id_to_class=raw_id_to_class_map,
            defaults=defaults,
            infos_prefix='scannet200_oneformer3d_infos',
            sem_id_to_label=sem_id_to_label,
            map_oneformer_labels=map_oneformer_labels,
            keep_background_default=False
        )

    if dataset_key == 'spp':
        defaults = scannetpp.defaults()
        if data_root is None:
            data_root = defaults['data_root']
        label_allowlist = label_allowlist or defaults.get('label_allowlist')
        allowlist_path = scannetpp.resolve_allowlist_path(data_root, label_allowlist)
        if allowlist_path is None:
            raise RuntimeError(
                f'Failed to resolve {scannetpp.ALLOWLIST_FILENAME}. Provide --label-allowlist with a valid path '
                f'or place {scannetpp.ALLOWLIST_FILENAME} under data_root/metadata/semantic_benchmark/.')
        if allowlist_path and osp.basename(allowlist_path) != scannetpp.ALLOWLIST_FILENAME:
            raise RuntimeError(
                f'Only {scannetpp.ALLOWLIST_FILENAME} is supported for this script (got: {allowlist_path}).')
        class_names = scannetpp.load_allowlist(allowlist_path)
        if not class_names:
            raise RuntimeError(f'Allowlist is empty: {allowlist_path}')
        raw_id_to_class_map = scannetpp.raw_id_to_class(class_names)
        stats = scannetpp.init_stats()

        def sem_id_to_label(raw_id: int) -> Optional[str]:
            return scannetpp.sem_id_to_label(raw_id, class_names, raw_id_to_class_map)

        def update_stats(sem_id: int, used: bool) -> None:
            scannetpp.update_stats(stats, sem_id, used)

        def finalize_run_cfg(run_cfg: Dict[str, object]) -> None:
            scannetpp.finalize_run_cfg(run_cfg, stats, allowlist_path, len(class_names))

        def map_oneformer_labels_spp(inst_labels: np.ndarray) -> np.ndarray:
            return scannetpp.map_oneformer_labels(inst_labels)

        return DatasetSpec(
            key='spp',
            class_names=class_names,
            raw_id_to_class=raw_id_to_class_map,
            defaults=defaults,
            infos_prefix='scannetpp_oneformer3d_infos',
            sem_id_to_label=sem_id_to_label,
            map_oneformer_labels=map_oneformer_labels_spp,
            stats=stats,
            update_stats=update_stats,
            finalize_run_cfg=finalize_run_cfg,
            allowlist_path=allowlist_path,
            keep_background_default=True
        )

    raise ValueError(f'Unsupported dataset: {dataset_key}')
