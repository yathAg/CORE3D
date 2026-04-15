#!/usr/bin/env python3
"""OneFormer3D graph builder."""

import json
import os.path as osp
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph.oneformer_mapping import match_preds_to_gt, PRED_MATCH_IOU_THR
from graph.graph_edges import REL_NAMES, build_relationships
from graph.graph_features import (
    apply_axis_align,
    centroid_from_mask,
    iqr_mean_color_from_values,
    majority_color_from_values,
    name_basic11,
    rotation_from_ypr,
)
from graph.graph_io import load_all_infos, load_points
from graph.graph_output import finalize_latency, init_output, write_common_outputs


def build_scene_graph_oneformer(scene_id: str,
                                matched: Dict[int, Dict],
                                points: np.ndarray,
                                axis_align: Optional[np.ndarray],
                                knn_k: Optional[int],
                                direction_plane: str,
                                use_cardinals: bool,
                                observer_rot: Optional[np.ndarray],
                                dataset) -> Tuple[Dict, Dict, Dict]:
    objects: Dict[int, Dict] = {}
    centroids: Dict[int, np.ndarray] = {}
    node_ids = sorted(matched.keys())

    xyz = points[:, :3]
    colors = points[:, 3:6]
    colors_int = np.clip(np.rint(colors), 0, 255).astype(np.int32)
    if axis_align is not None:
        xyz = apply_axis_align(xyz, axis_align)

    for gid in node_ids:
        m = matched[gid]
        mask = m['mask']
        centroid = centroid_from_mask(xyz, mask)
        if centroid is None:
            continue
        color_slice = colors_int[mask]
        iqr_color = iqr_mean_color_from_values(color_slice)
        maj_color = majority_color_from_values(color_slice)
        if iqr_color is None:
            continue
        label_id = int(m['label_id'])
        label = dataset.sem_id_to_label(label_id)
        if label is None:
            continue
        size = int(mask.sum())
        color_name = name_basic11(iqr_color)
        objects[int(gid)] = {
            'label': label,
            'label_id': label_id,
            'score': float(m['score']),
            'orig_id': int(gid),
            'size': size,
            'centroid': centroid,
            'majority_color': maj_color,
            'iqr_color': iqr_color,
            'color_name': color_name,
            'color_source': 'iqr_mean'
        }
        centroids[int(gid)] = np.array(centroid, dtype=np.float32)

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


def run_oneformer(args, dataset) -> None:
    if dataset.map_oneformer_labels is None:
        raise NotImplementedError(f'OneFormer graphs not implemented for dataset={dataset.key}')

    import torch

    infos, _ = load_all_infos(args.data_root, dataset.infos_prefix)

    scenes_dir, relationships_all = init_output(args.out_root)

    latency_scene_count = 0
    latency_total_sum = 0.0
    latency_breakdown_sum: Dict[str, float] = {}
    latency_breakdown_count: Dict[str, int] = {}
    scan_list: List[str] = []

    results_dir = Path(args.results_dir)
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
        pred_path = results_dir / f'{scene_id}.pth'
        if not pred_path.exists():
            continue

        t_scene_start = time.perf_counter() if args.profile_latency else None
        breakdown: Dict[str, float] = {}

        t0 = time.perf_counter()
        gt_inst = np.fromfile(osp.join(args.data_root, 'instance_mask', inst_rel), dtype=np.int64)
        gt_sem = np.fromfile(osp.join(args.data_root, 'semantic_mask', sem_rel), dtype=np.int64)
        if args.profile_latency:
            breakdown['load_masks_sec'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_obj = torch.load(pred_path, map_location='cpu', weights_only=False)
        inst_masks = np.asarray(pred_obj.get('instance_masks'))
        inst_labels = np.asarray(pred_obj.get('instance_labels'))
        inst_scores = np.asarray(pred_obj.get('instance_scores'))
        if args.profile_latency:
            breakdown['load_predictions_sec'] = time.perf_counter() - t0
        if inst_masks.size == 0:
            continue

        inst_labels_raw = dataset.map_oneformer_labels(inst_labels)

        sample_indices = pred_obj.get('sample_indices')
        t0 = time.perf_counter()
        if sample_indices is not None:
            sample_indices = np.asarray(sample_indices, dtype=np.int64)
            n = min(inst_masks.shape[1], sample_indices.shape[0])
            sample_indices = sample_indices[:n]
            gt_inst = gt_inst[sample_indices]
            gt_sem = gt_sem[sample_indices]
            inst_masks = inst_masks[:, :n]
            points = load_points(args.data_root, points_rel)[sample_indices]
        else:
            n = min(len(gt_inst), len(gt_sem), inst_masks.shape[1])
            gt_inst = gt_inst[:n]
            gt_sem = gt_sem[:n]
            inst_masks = inst_masks[:, :n]
            points = load_points(args.data_root, points_rel)[:n]
        if args.profile_latency:
            breakdown['load_points_sec'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        matched = match_preds_to_gt(inst_masks, inst_labels_raw, inst_scores,
                                    gt_inst, gt_sem, iou_thr=PRED_MATCH_IOU_THR)
        if args.profile_latency:
            breakdown['match_preds_sec'] = time.perf_counter() - t0
        if not matched:
            continue

        axis_align = None
        if args.enable_axis_align:
            axis_align = np.array(info.get('axis_align_matrix', np.eye(4)), dtype=np.float32)

        t0 = time.perf_counter()
        scene_graph, neighbors, edge_features = build_scene_graph_oneformer(
            scene_id, matched, points, axis_align, args.knn,
            args.direction_plane, args.use_cardinals, observer_rot, dataset)
        if args.profile_latency:
            breakdown['build_scene_graph_sec'] = time.perf_counter() - t0

        scene_graph['neighbors'] = neighbors
        scene_graph['ids_aligned_to_gt'] = True
        scene_graph['id_source'] = 'gt_instance_id'
        scene_graph['matching_iou_thresh'] = PRED_MATCH_IOU_THR
        scene_graph['prediction_source'] = 'oneformer3d'
        scene_graph['pred_alignment'] = {
            int(gid): {
                'pred_idx': int(info['pred_idx']),
                'iou': float(info['iou']),
                'score': float(info['score']),
            }
            for gid, info in matched.items()
        }

        relationships_all['scans'].append(scene_graph)
        relationships_all['neighbors'][scene_id] = neighbors
        relationships_all['edge_features'][scene_id] = edge_features
        scan_list.append(scene_id)

        out_scene_path = osp.join(scenes_dir, f'{scene_id}.json')
        t0 = time.perf_counter()
        with open(out_scene_path, 'w') as f:
            json.dump(scene_graph, f)
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
    run_cfg['ids_aligned_to_gt'] = True
    run_cfg['matching_iou_thresh'] = PRED_MATCH_IOU_THR
    run_cfg['prediction_source'] = 'oneformer3d'

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
