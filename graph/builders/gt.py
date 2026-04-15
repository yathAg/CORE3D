#!/usr/bin/env python3
"""GT graph builder."""

import json
import os.path as osp
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph.graph_edges import REL_NAMES, build_relationships
from graph.graph_features import (
    apply_axis_align,
    centroid_from_points,
    iqr_mean_color_from_values,
    majority_color_from_values,
    majority_label_from_values,
    name_basic11,
    rotation_from_ypr,
)
from graph.graph_io import load_all_infos, load_masks, load_points
from graph.graph_output import finalize_latency, init_output, write_common_outputs


def build_scene_graph_gt(scene_id: str,
                         inst_mask: np.ndarray,
                         sem_mask: np.ndarray,
                         points: np.ndarray,
                         axis_align: Optional[np.ndarray],
                         knn_k: Optional[int],
                         keep_background: bool,
                         direction_plane: str,
                         use_cardinals: bool,
                         observer_rot: Optional[np.ndarray],
                         dataset) -> Tuple[Dict, Dict, Dict]:
    objects: Dict[int, Dict] = {}
    centroids: Dict[int, np.ndarray] = {}

    xyz = points[:, :3]
    if axis_align is not None:
        xyz = apply_axis_align(xyz, axis_align)
    colors = points[:, 3:6]
    colors_int = np.clip(np.rint(colors), 0, 255).astype(np.int32)

    if inst_mask.size > 0:
        order = np.argsort(inst_mask, kind='mergesort')
        inst_sorted = inst_mask[order]
        sem_sorted = sem_mask[order]
        xyz_sorted = xyz[order]
        colors_sorted = colors_int[order]
        uniq_ids, starts, counts = np.unique(inst_sorted, return_index=True, return_counts=True)
    else:
        uniq_ids, starts, counts = np.array([], dtype=inst_mask.dtype), np.array([], dtype=int), np.array([], dtype=int)

    for orig_id, start, count in zip(uniq_ids, starts, counts):
        if orig_id == 0 and not keep_background:
            continue
        if orig_id < 0:
            continue
        end = int(start + count)
        sem_slice = sem_sorted[start:end]
        sem_id = majority_label_from_values(sem_slice)
        if sem_id is None:
            continue
        label = dataset.sem_id_to_label(int(sem_id))
        if dataset.update_stats is not None:
            dataset.update_stats(int(sem_id), label is not None)
        if label is None:
            continue
        size = int(count)
        color_slice = colors_sorted[start:end]
        maj_color = majority_color_from_values(color_slice)
        iqr_color = iqr_mean_color_from_values(color_slice)
        centroid = centroid_from_points(xyz_sorted[start:end])
        if iqr_color is None or centroid is None:
            continue
        color_name = name_basic11(iqr_color)
        objects[int(orig_id)] = {
            'label': label,
            'label_id': int(sem_id),
            'score': 1.0,
            'orig_id': int(orig_id),
            'size': size,
            'centroid': centroid,
            'majority_color': maj_color,
            'iqr_color': iqr_color,
            'color_name': color_name,
            'color_source': 'iqr_mean'
        }
        centroids[int(orig_id)] = np.array(centroid, dtype=np.float32)

    node_ids = sorted(objects.keys())
    relationships, neighbors, edge_features = build_relationships(
        node_ids, centroids, knn_k, direction_plane, use_cardinals, observer_rot)

    scene_entry = {
        'scan': scene_id,
        'split': 0,
        'objects': {k: v['label'] for k, v in objects.items()},
        'relationships': relationships,
        'edge_features': edge_features,
        'object_metadata': objects
    }
    return scene_entry, neighbors, edge_features


def run_gt(args, dataset) -> None:
    infos, _ = load_all_infos(args.data_root, dataset.infos_prefix)

    scenes_dir, relationships_all = init_output(args.out_root)

    latency_scene_count = 0
    latency_total_sum = 0.0
    latency_breakdown_sum: Dict[str, float] = {}
    latency_breakdown_count: Dict[str, int] = {}
    scan_list: List[str] = []

    seen = set()
    observer_rot = None
    if args.enable_observer_frame:
        yaw, pitch, roll = args.observer_yaw_pitch_roll
        observer_rot = rotation_from_ypr(yaw, pitch, roll)

    for info in infos:
        if 'pts_instance_mask_path' not in info or 'pts_semantic_mask_path' not in info:
            continue
        inst_rel = info['pts_instance_mask_path']
        sem_rel = info['pts_semantic_mask_path']
        points_rel = info.get('lidar_points', {}).get('lidar_path')
        if points_rel is None:
            continue
        scene_id = osp.splitext(osp.basename(inst_rel))[0]
        if scene_id in seen:
            continue
        seen.add(scene_id)

        t_scene_start = time.perf_counter() if args.profile_latency else None
        breakdown: Dict[str, float] = {}

        t0 = time.perf_counter()
        inst_mask, sem_mask = load_masks(args.data_root, inst_rel, sem_rel)
        if args.profile_latency:
            breakdown['load_masks_sec'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        points = load_points(args.data_root, points_rel)
        if args.profile_latency:
            breakdown['load_points_sec'] = time.perf_counter() - t0

        axis_align = None
        if args.enable_axis_align:
            axis_align = np.array(info.get('axis_align_matrix', np.eye(4)), dtype=np.float32)

        t0 = time.perf_counter()
        scene_graph, neighbors, edge_features = build_scene_graph_gt(
            scene_id, inst_mask, sem_mask, points, axis_align,
            args.knn, args.keep_background, args.direction_plane,
            args.use_cardinals, observer_rot, dataset)
        if args.profile_latency:
            breakdown['build_scene_graph_sec'] = time.perf_counter() - t0

        relationships_all['scans'].append(scene_graph)
        relationships_all['neighbors'][scene_id] = neighbors
        relationships_all['edge_features'][scene_id] = edge_features
        scan_list.append(scene_id)

        t0 = time.perf_counter()
        out_scene_path = osp.join(scenes_dir, f'{scene_id}.json')
        with open(out_scene_path, 'w') as f:
            json.dump({
                'scan': scene_id,
                'objects': scene_graph['objects'],
                'relationships': scene_graph['relationships'],
                'edge_features': edge_features,
                'neighbors': neighbors,
                'object_metadata': scene_graph['object_metadata']
            }, f)
        if args.profile_latency:
            breakdown['write_scene_json_sec'] = time.perf_counter() - t0

        if args.profile_latency and t_scene_start is not None:
            latency_scene_count += 1
            latency_total_sum += time.perf_counter() - t_scene_start
            for key, val in breakdown.items():
                latency_breakdown_sum[key] = latency_breakdown_sum.get(key, 0.0) + val
                latency_breakdown_count[key] = latency_breakdown_count.get(key, 0) + 1

    run_cfg = vars(args)
    run_cfg['relation_names'] = REL_NAMES
    run_cfg['save_per_scene'] = True
    run_cfg['splits_processed'] = ['train', 'val', 'test']
    if dataset.finalize_run_cfg is not None:
        dataset.finalize_run_cfg(run_cfg)

    latency_stats = finalize_latency(
        args.profile_latency,
        latency_scene_count,
        latency_total_sum,
        latency_breakdown_sum,
        latency_breakdown_count)

    write_common_outputs(args.out_root, relationships_all, dataset.class_names, scan_list, run_cfg, latency_stats)

    if args.profile_latency and latency_stats is not None:
        print('Latency profiling (seconds):')
        print(f"  total_scenes={latency_stats['total_scenes']}, avg_total={latency_stats['avg_total_sec']:.3f}s")
        if latency_stats['avg_breakdown_sec']:
            parts = [f"{k}={v:.3f}" for k, v in latency_stats['avg_breakdown_sec'].items()]
            print(f"  avg_breakdown: {', '.join(parts)}")

    print(f'Done. Processed {len(scan_list)} scenes. Outputs at {args.out_root}')
