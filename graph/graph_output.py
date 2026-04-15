#!/usr/bin/env python3
"""Output helpers for graph building."""

import json
import os
import os.path as osp
from typing import Dict, List, Optional

from graph.graph_edges import REL_NAMES


def init_output(out_root: str):
    scenes_dir = osp.join(out_root, 'scenes')
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)
    relationships_all = {'scans': [], 'neighbors': {}, 'edge_features': {}}
    return scenes_dir, relationships_all


def write_common_outputs(out_root: str,
                         relationships_all: Dict,
                         class_names: List[str],
                         scan_list: List[str],
                         run_cfg: Dict,
                         latency_stats: Optional[Dict]):
    with open(osp.join(out_root, 'relationships.json'), 'w') as f:
        json.dump(relationships_all, f)

    with open(osp.join(out_root, 'classes.txt'), 'w') as f:
        for name in class_names:
            f.write(f'{name}\n')

    with open(osp.join(out_root, 'relationships.txt'), 'w') as f:
        for rel in REL_NAMES:
            f.write(f'{rel}\n')

    with open(osp.join(out_root, 'scans.txt'), 'w') as f:
        for sid in scan_list:
            f.write(f'{sid}\n')

    with open(osp.join(out_root, 'run_args.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2)

    if latency_stats is not None:
        with open(osp.join(out_root, 'latency.json'), 'w') as f:
            json.dump(latency_stats, f, indent=2)


def finalize_latency(profile_latency: bool,
                     latency_scene_count: int,
                     latency_total_sum: float,
                     latency_breakdown_sum: Dict[str, float],
                     latency_breakdown_count: Dict[str, int]) -> Optional[Dict]:
    if not profile_latency:
        return None
    avg_total = latency_total_sum / latency_scene_count if latency_scene_count else 0.0
    avg_breakdown = {
        key: latency_breakdown_sum[key] / latency_breakdown_count[key]
        for key in latency_breakdown_sum
        if latency_breakdown_count.get(key, 0) > 0
    }
    return {
        'total_scenes': latency_scene_count,
        'avg_total_sec': avg_total,
        'avg_breakdown_sec': avg_breakdown
    }
