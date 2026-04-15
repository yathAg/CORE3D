#!/usr/bin/env python
"""Generate EZ-SP superpoint bins (OneFormer-style) from raw ScanNet data.

Example:
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m ezsp.generate_superpoints \
    --raw_root data/scannet200 \
    --ckpt_path /path/to/partition_ckpt.ckpt \
    --output_dir output/super_points_ezsp
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import os.path as osp
import sys
import time
from typing import List

import numpy as np
import torch

try:
    import torch.cuda.nvtx as nvtx
    _NVTX_AVAILABLE = True
except Exception:
    nvtx = None
    _NVTX_AVAILABLE = False

EZSP_ROOT = osp.dirname(osp.abspath(__file__))
if EZSP_ROOT not in sys.path:
    sys.path.insert(0, EZSP_ROOT)

from src.datasets.scannet import read_scannet_scan, resolve_label_map_file
from src.datasets.scannet_config import SCANS, SCANNET200_NUM_CLASSES
from src.transforms.device import DataTo
from src.transforms.sampling import SaveNodeIndex, GridSampling3D
from src.transforms.point import PointFeatures, PretrainedCNN, GroundElevation
from src.transforms.neighbors import KNN
from src.transforms.graph import AdjacencyGraph
from src.transforms.partition import GreedyContourPriorPartition
from src.nn import GraphNorm
from src.nn.stage import PointStage
from src.utils.components_merge import merge_components_by_contour_prior_on_data
from src.data import Data
from src.utils.scannet import read_mesh_vertices, read_axis_align_matrix
from src.utils.color import to_float_rgb


def _parse_list(values: str, cast_fn):
    if values is None:
        return None
    parts = [p.strip() for p in values.split(',') if p.strip() != '']
    return [cast_fn(p) for p in parts]


def _parse_reg(values: str):
    if values is None:
        return None
    if ',' in values:
        return _parse_list(values, float)
    return float(values)


# Feature size mapping (from configs/datamodule/semantic/_features.yaml)
FEAT_SIZE = {
    'pos': 3,
    'pos_room': 3,
    'rgb': 3,
    'hsv': 3,
    'lab': 3,
    'density': 1,
    'linearity': 1,
    'planarity': 1,
    'scattering': 1,
    'verticality': 1,
    'normal': 3,
    'length': 1,
    'surface': 1,
    'volume': 1,
    'curvature': 1,
    'elevation': 1,
    'size': 1,
    'intensity': 1,
    'log_pos': 3,
    'log_pos_room': 3,
    'log_rgb': 3,
    'log_hsv': 3,
    'log_lab': 3,
    'log_density': 1,
    'log_linearity': 1,
    'log_planarity': 1,
    'log_scattering': 1,
    'log_verticality': 1,
    'log_normal': 3,
    'log_length': 1,
    'log_surface': 1,
    'log_volume': 1,
    'log_curvature': 1,
    'log_elevation': 1,
    'log_size': 1,
    'mean_pos': 3,
    'mean_pos_room': 3,
    'mean_rgb': 3,
    'mean_hsv': 3,
    'mean_lab': 3,
    'mean_density': 1,
    'mean_linearity': 1,
    'mean_planarity': 1,
    'mean_scattering': 1,
    'mean_verticality': 1,
    'mean_normal': 3,
    'mean_length': 1,
    'mean_surface': 1,
    'mean_volume': 1,
    'mean_curvature': 1,
    'mean_elevation': 1,
    'mean_size': 1,
    'mean_intensity': 1,
    'std_pos': 3,
    'std_pos_room': 3,
    'std_rgb': 3,
    'std_hsv': 3,
    'std_lab': 3,
    'std_density': 1,
    'std_linearity': 1,
    'std_planarity': 1,
    'std_scattering': 1,
    'std_verticality': 1,
    'std_normal': 3,
    'std_length': 1,
    'std_surface': 1,
    'std_volume': 1,
    'std_curvature': 1,
    'std_elevation': 1,
    'std_size': 1,
    'std_intensity': 1,
    'mean_off': 3,
    'std_off': 3,
    'mean_dist': 1,
    'angle_source': 1,
    'angle_target': 1,
    'centroid_dir': 3,
    'centroid_dist': 1,
    'normal_angle': 1,
}


def _now_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _resolve_raw_dir(raw_root: str) -> str:
    raw_root = osp.normpath(raw_root)
    raw_tail = osp.basename(raw_root)
    if raw_tail in ("scans", "scans_test"):
        return osp.dirname(raw_root)
    return raw_root


def _cache_root_with_hash(base_root: str, params: dict, auto: bool) -> str:
    if not auto:
        return base_root
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return osp.join(base_root, f"auto_{digest}")


def _init_run_dir(runs_root: str, run_dir: str) -> str:
    final_dir = run_dir or osp.join(runs_root, _now_timestamp())
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def build_first_stage(num_hf_partition: int,
                      cnn_dim_without_in_dim: List[int],
                      kernel_size: int,
                      dilation: int):
    cnn = [num_hf_partition] + cnn_dim_without_in_dim
    return PointStage(
        in_mlp=None,
        use_pos=False,
        use_diameter_parent=False,
        cnn_blocks=True,
        cnn=cnn,
        cnn_kernel_size=kernel_size,
        cnn_dilation=dilation,
        cnn_norm=GraphNorm,
        cnn_activation=torch.nn.LeakyReLU(),
        cnn_residual=False,
        cnn_global_residual=False,
    ) 


def needs_knn_features(partition_hf: List[str]) -> bool:
    if partition_hf is None:
        return False
    knn_keys = {
        'density', 'linearity', 'planarity', 'scattering', 'verticality',
        'normal', 'length', 'surface', 'volume', 'curvature'
    }
    return any(k in knn_keys for k in partition_hf)


def scene_id_from_rel(rel_path: str) -> str:
    return osp.basename(rel_path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_raw_path(raw_root: str, rel_path: str) -> str:
    """Resolve a raw scan path.

    Supports raw_root pointing to either:
      - .../raw (contains scans/ and scans_test/)
      - .../raw/scans or .../raw/scans_test (contains scene folders)
    """
    rel_path = osp.normpath(rel_path)
    rel_parts = rel_path.split(os.sep)
    if len(rel_parts) < 2:
        return osp.join(raw_root, rel_path)

    rel_root = rel_parts[0]  # 'scans' or 'scans_test'
    scene_id = rel_parts[-1]

    raw_root = osp.normpath(raw_root)
    raw_tail = osp.basename(raw_root)
    if raw_tail in ("scans", "scans_test"):
        if raw_tail == rel_root:
            return osp.join(raw_root, scene_id)
        return osp.join(osp.dirname(raw_root), rel_path)

    return osp.join(raw_root, rel_path)


def _apply_axis_alignment(pos: torch.Tensor, meta_file: str) -> torch.Tensor:
    axis_align_matrix = read_axis_align_matrix(meta_file)
    if axis_align_matrix is None:
        return pos
    axis_align_matrix = axis_align_matrix.to(pos.device)
    pts = torch.ones((pos.shape[0], 4), dtype=pos.dtype, device=pos.device)
    pts[:, 0:3] = pos
    pts = torch.matmul(pts, axis_align_matrix.t())
    return pts[:, 0:3]


@contextlib.contextmanager
def nvtx_range(name: str):
    if _NVTX_AVAILABLE:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if _NVTX_AVAILABLE:
            nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser(description="Generate EZ-SP superpoint bins from raw ScanNet data")
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Path to ScanNet raw root (contains scans/ and scans_test/) OR directly to raw/scans",
    )
    parser.add_argument("--label_map_file", default=None,
                        help="Optional path to scannetv2-labels.combined.tsv (auto-resolved if omitted)")
    parser.add_argument("--axis_align", dest="axis_align", action="store_true", default=False,
                        help="Apply axis alignment to raw ScanNet points")
    parser.add_argument("--no_axis_align", dest="axis_align", action="store_false",
                        help="Skip axis alignment (use raw ScanNet coordinates; default)")
    parser.add_argument("--ckpt_path", required=True, help="Path to pretrained EZ-SP CNN checkpoint")
    parser.add_argument(
        "--output_dir",
        default=osp.join(EZSP_ROOT, "output", "super_points_ezsp"),
        help="Output directory for .bin superpoint files (inside ezsp)",
    )
    parser.add_argument("--runs_root", default=osp.join(EZSP_ROOT, "runs", "ezsp_superpoints"),
                        help="Root directory for per-run logs/configs")
    parser.add_argument("--run_dir", default=None,
                        help="Optional explicit run directory (overrides runs_root)")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to run")
    parser.add_argument("--device", default="cuda", help="Device for preprocessing/partition")
    parser.add_argument(
        "--with_labels",
        dest="with_labels",
        action="store_true",
        default=False,
        help="Load semantic labels (slower). Default: off.",
    )

    # Cache
    parser.add_argument(
        "--cache_root",
        default=osp.join(EZSP_ROOT, "cache", "scannet200_partition"),
        help="Cache directory for preprocessed data (inside ezsp)",
    )
    parser.add_argument("--cache_auto", dest="cache_auto", action="store_true", default=True,
                        help="Append a hash of preprocessing params to cache_root (default)")
    parser.add_argument("--cache_legacy", dest="cache_auto", action="store_false",
                        help="Use cache_root as-is (no param hash)")
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_use_cache", dest="use_cache", action="store_false")
    parser.add_argument("--write_cache", action="store_true", default=True)
    parser.add_argument("--no_write_cache", dest="write_cache", action="store_false")

    # EZ-SP / partition parameters (defaults match scannet200_ezsp config)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--quantize_coords", action="store_true", default=True)
    parser.add_argument("--no_quantize_coords", action="store_false", dest="quantize_coords")

    parser.add_argument("--partition_hf", default="rgb", help="Comma-separated point features for CNN input")

    parser.add_argument("--cnn_dim_without_in", default="32,32,32", help="CNN channel dims excluding input")
    parser.add_argument("--cnn_kernel_size", type=int, default=3)
    parser.add_argument("--cnn_dilation", type=int, default=1)

    parser.add_argument("--contour_reg", default="0.02", help="Partition reg (float or comma list)")
    parser.add_argument("--contour_min_size", default="5,30,90", help="Comma-separated min_size per level")
    parser.add_argument("--contour_edge_mode", default="unit")
    parser.add_argument("--contour_edge_reduce", default="add")
    parser.add_argument("--contour_k_isolated", type=int, default=0)
    parser.add_argument("--contour_sharding", type=float, default=None)

    parser.add_argument("--contour_knn", type=int, default=8)
    parser.add_argument("--contour_knn_r", type=float, default=2.0)
    parser.add_argument(
        "--contour_max_iterations",
        type=int,
        default=-1,
        help="Max merge iterations for contour prior (<=0 means auto).",
    )
    parser.add_argument(
        "--contour_merge_only_small",
        action="store_true",
        default=False,
        help="Only merge components below min_size (skip energy-based merges).",
    )

    parser.add_argument("--strict_config", action="store_true", default=False,
                        help="Error on checkpoint/CLI config mismatch (default: warn)")

    # Ground elevation (optional)
    parser.add_argument("--ground_model", default=None,
                        help="Ground model for elevation (e.g., ransac). If None, skip.")
    parser.add_argument("--ground_threshold", type=float, default=0.05)
    parser.add_argument("--ground_xy_grid", type=float, default=0.2)
    parser.add_argument("--ground_scale", type=float, default=1.0)

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .bin outputs")
    parser.add_argument("--limit_scenes", type=int, default=None,
                        help="Process at most N scenes per split (dry-run helper).")
    parser.add_argument("--only_scene", default=None,
                        help="Process only this scene id (overrides --limit_scenes).")
    parser.add_argument(
        "--flat_output",
        dest="flat_output",
        action="store_true",
        default=True,
        help="Write all .bin files directly under output_dir (default).",
    )
    parser.add_argument(
        "--split_output",
        dest="flat_output",
        action="store_false",
        help="Write .bin files into output_dir/<split>/ subfolders.",
    )

    parser.add_argument(
        "--print_per_scene",
        dest="print_per_scene",
        action="store_true",
        default=True,
        help="Print timing per scene (default).",
    )
    parser.add_argument(
        "--no_print_per_scene",
        dest="print_per_scene",
        action="store_false",
        help="Disable per-scene timing prints.",
    )

    args = parser.parse_args()

    run_dir = _init_run_dir(args.runs_root, args.run_dir)
    log_path = osp.join(run_dir, "run.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[run] dir={run_dir}")

    splits = [s.strip() for s in args.splits.split(',') if s.strip() != '']
    partition_hf = _parse_list(args.partition_hf, str)
    cnn_dim_without_in = _parse_list(args.cnn_dim_without_in, int)
    contour_min_size = _parse_list(args.contour_min_size, int)
    contour_reg = _parse_reg(args.contour_reg)

    if contour_reg is None:
        contour_reg = 0.02
    if contour_min_size is None:
        contour_min_size = [5, 30, 90]

    # Single-level partitioning only (use level-0 settings for flat output)
    if isinstance(contour_min_size, list):
        if len(contour_min_size) > 1:
            print(f"[info] Using single-level partition: min_size={contour_min_size[0]} "
                  f"(discarding {contour_min_size[1:]})")
        contour_min_size = contour_min_size[0]
    if isinstance(contour_reg, list):
        if len(contour_reg) > 1:
            print(f"[info] Using single-level partition: reg={contour_reg[0]} "
                  f"(discarding {contour_reg[1:]})")
        contour_reg = contour_reg[0]

    device = torch.device(args.device)
    num_classes = SCANNET200_NUM_CLASSES

    missing = [k for k in partition_hf if k not in FEAT_SIZE]
    if missing:
        raise ValueError(f"Unknown feature(s) in --partition_hf: {missing}")

    if not args.quantize_coords:
        raise ValueError(
            "quantize_coords=False is incompatible with the sparse CNN "
            "(PointStage uses torchsparse). Keep quantize_coords enabled."
        )

    raw_dir = _resolve_raw_dir(args.raw_root)
    label_map_file = args.label_map_file or resolve_label_map_file(raw_dir)

    cache_params = {
        "dataset": "scannet200",
        "voxel": args.voxel,
        "quantize_coords": args.quantize_coords,
        "partition_hf": partition_hf,
        "axis_align": args.axis_align,
        "label_type": "scannet200",
        "label_map_file": label_map_file,
        "ground_model": args.ground_model,
        "ground_threshold": args.ground_threshold,
        "ground_xy_grid": args.ground_xy_grid,
        "ground_scale": args.ground_scale,
        "knn_k": args.contour_knn,
        "knn_r": args.contour_knn_r,
        "with_labels": args.with_labels,
    }
    args.cache_root = _cache_root_with_hash(args.cache_root, cache_params, args.cache_auto)
    os.makedirs(args.cache_root, exist_ok=True)
    _write_json(osp.join(args.cache_root, "cache_meta.json"), cache_params)

    run_config = {
        "timestamp": _now_timestamp(),
        "args": vars(args),
        "resolved": {
            "run_dir": run_dir,
            "log_path": log_path,
            "cache_root": args.cache_root,
            "label_map_file": label_map_file,
        },
        "cache_params": cache_params,
    }
    _write_json(osp.join(run_dir, "run_config.json"), run_config)

    ckpt_meta = {}
    try:
        ckpt_blob = torch.load(args.ckpt_path, map_location="cpu")
        ckpt_meta = ckpt_blob.get("metadata", {}) if isinstance(ckpt_blob, dict) else {}
    except Exception as e:
        print(f"[warn] Could not read checkpoint metadata: {e}")

    def _compare_meta(key, expected):
        if key in ckpt_meta and ckpt_meta.get(key) != expected:
            msg = f"[config] mismatch for {key}: ckpt={ckpt_meta.get(key)} vs args={expected}"
            if args.strict_config:
                raise ValueError(msg)
            print(f"[warn] {msg}")

    _compare_meta("partition_hf", partition_hf)
    _compare_meta("cnn_dim_without_in", cnn_dim_without_in)
    _compare_meta("cnn_kernel_size", args.cnn_kernel_size)
    _compare_meta("cnn_dilation", args.cnn_dilation)
    _compare_meta("voxel", args.voxel)
    _compare_meta("quantize_coords", args.quantize_coords)
    _compare_meta("axis_align", args.axis_align)
    _compare_meta("ground_model", args.ground_model)

    num_hf_partition = sum(FEAT_SIZE[k] for k in partition_hf)

    # Build transforms
    print("Building transforms...")
    data_to = DataTo(device=device)
    save_node_index = SaveNodeIndex(key='sub')
    grid = GridSampling3D(
        size=args.voxel,
        quantize_coords=args.quantize_coords,
        hist_key='y' if args.with_labels else None,
        hist_size=(num_classes + 1) if args.with_labels else None,
    )
    point_features = PointFeatures(keys=partition_hf, overwrite=False) if partition_hf else None

    knn = KNN(k=args.contour_knn, r_max=args.contour_knn_r,
              verbose=False, self_is_neighbor=False, save_as_csr=False)
    adj = AdjacencyGraph(k=args.contour_knn, w=-1)

    first_stage = build_first_stage(
        num_hf_partition=num_hf_partition,
        cnn_dim_without_in_dim=cnn_dim_without_in,
        kernel_size=args.cnn_kernel_size,
        dilation=args.cnn_dilation,
    )

    print(f"Loading checkpoint: {args.ckpt_path}")
    pretrained = PretrainedCNN(
        first_stage=first_stage,
        ckpt_path=args.ckpt_path,
        partition_hf=partition_hf,
        norm_mode='graph',
        device=device,
        verbose=False,
    )
    pretrained.first_stage.eval()

    partition_helper = GreedyContourPriorPartition(
        reg=contour_reg,
        min_size=contour_min_size,
        edge_weight_mode=args.contour_edge_mode,
        edge_reduce=args.contour_edge_reduce,
        k=args.contour_k_isolated,
        w_adjacency=0.0,
        max_iterations=args.contour_max_iterations,
        verbose=False,
        sharding=args.contour_sharding,
    )

    # Output dirs
    if args.flat_output:
        ensure_dir(args.output_dir)
    else:
        for split in splits:
            ensure_dir(osp.join(args.output_dir, split))

    torch.set_grad_enabled(False)
    total_compute_time = 0.0
    total_load_time = 0.0
    total_save_time = 0.0
    total_superpoints = 0
    total_scenes = 0

    for split in splits:
        if split not in SCANS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(SCANS.keys())}")

        scene_list = SCANS[split]
        if args.only_scene is not None:
            scene_list = [rel for rel in scene_list
                          if scene_id_from_rel(rel) == args.only_scene]
        elif args.limit_scenes is not None:
            scene_list = scene_list[:args.limit_scenes]
        for rel in scene_list:
            scene_id = scene_id_from_rel(rel)
            if args.flat_output:
                out_path = osp.join(args.output_dir, f"{scene_id}.bin")
            else:
                out_path = osp.join(args.output_dir, split, f"{scene_id}.bin")
            if (not args.overwrite) and osp.exists(out_path):
                if args.print_per_scene:
                    print(f"[skip] {scene_id} (exists)")
                continue

            cache_path = osp.join(args.cache_root, split, f"{scene_id}.pt")
            data = None
            if args.use_cache and osp.exists(cache_path):
                t_load = time.perf_counter()
                data = torch.load(cache_path)
                load_elapsed = time.perf_counter() - t_load
                total_load_time += load_elapsed

            if data is None:
                raw_path = resolve_raw_path(args.raw_root, rel)
                if not osp.exists(raw_path):
                    print(f"[skip] missing raw path: {raw_path}")
                    continue
                if args.print_per_scene:
                    print(f"[start] {scene_id}")

                t_load = time.perf_counter()
                with nvtx_range("load_scan"):
                    if args.with_labels:
                        data = read_scannet_scan(
                            raw_path,
                            xyz=True,
                            rgb=True,
                            normal=False,
                            semantic=True,
                            instance=False,
                            remap=True,
                            label_type="scannet200",
                            label_map_file=label_map_file,
                            axis_align=args.axis_align,
                        )
                    else:
                        ply_path = osp.join(raw_path, f"{scene_id}_vh_clean_2.ply")
                        verts = read_mesh_vertices(ply_path, rgb=True, normal=False)
                        pos = torch.from_numpy(verts[:, :3])
                        if args.axis_align:
                            meta_file = osp.join(raw_path, f"{scene_id}.txt")
                            if osp.exists(meta_file):
                                pos = _apply_axis_alignment(pos, meta_file)
                        rgb = to_float_rgb(torch.from_numpy(verts[:, 3:6]))
                        data = Data(pos=pos, rgb=rgb)
                load_elapsed = time.perf_counter() - t_load
                total_load_time += load_elapsed

                # Preprocess (CPU -> GPU)
                with torch.inference_mode():
                    data = save_node_index(data)
                    data = grid(data)
                    if point_features is not None and not needs_knn_features(partition_hf):
                        data = point_features(data)
                    if args.ground_model is not None and 'elevation' in partition_hf:
                        data = GroundElevation(
                            model=args.ground_model,
                            z_threshold=args.ground_threshold,
                            xy_grid=args.ground_xy_grid,
                            scale=args.ground_scale,
                        )(data)

                data = data.to('cpu')
                if args.write_cache:
                    os.makedirs(osp.dirname(cache_path), exist_ok=True)
                    torch.save(data, cache_path)

            # Preprocess + partition (on GPU)
            _sync_device(device)
            t_compute = time.perf_counter()
            with torch.inference_mode():
                with nvtx_range("to_device"):
                    data = data_to(data)
                with nvtx_range("knn"):
                    data = knn(data)
                if point_features is not None and needs_knn_features(partition_hf):
                    with nvtx_range("point_features"):
                        data = point_features(data)
                    if args.ground_model is not None and 'elevation' in partition_hf:
                        data = GroundElevation(
                            model=args.ground_model,
                            z_threshold=args.ground_threshold,
                            xy_grid=args.ground_xy_grid,
                            scale=args.ground_scale,
                        )(data)
                with nvtx_range("adjacency"):
                    data = adj(data)
                with nvtx_range("pretrained_cnn"):
                    data = pretrained(data)

                with nvtx_range("edge_weights"):
                    data = partition_helper.edge_weights(data, data.edge_index)
                with nvtx_range("concat_pos"):
                    data = partition_helper.concatenate_pos_to_x(data)

                # Run contour-prior merge directly (skip NAG construction)
                with nvtx_range("partition"):
                    data, _ = merge_components_by_contour_prior_on_data(
                        data,
                        reg=contour_reg,
                        min_size=contour_min_size,
                        merge_only_small=args.contour_merge_only_small,
                        k=args.contour_k_isolated,
                        w_adjacency=0.0,
                        max_iterations=args.contour_max_iterations,
                        verbose=False,
                        sharding=args.contour_sharding,
                        reduce=args.contour_edge_reduce,
                    )

                # Map raw points -> voxel -> superpoint
                with nvtx_range("map_raw_to_superpoint"):
                    raw_to_voxel = data.sub.to_super_index()
                    raw_to_sp = data.super_index[raw_to_voxel]

            sp_np = raw_to_sp.detach().cpu().numpy().astype(np.int64)
            _sync_device(device)
            compute_elapsed = time.perf_counter() - t_compute
            total_compute_time += compute_elapsed

            t_save = time.perf_counter()
            with nvtx_range("save_bin"):
                sp_np.tofile(out_path)
            save_elapsed = time.perf_counter() - t_save
            total_save_time += save_elapsed

            total_scenes += 1
            if sp_np.size > 0:
                num_sp = int(sp_np.max()) + 1
                total_superpoints += num_sp
            else:
                num_sp = 0

            if args.print_per_scene:
                print(
                    f"[done] {scene_id} | compute={compute_elapsed:.3f}s "
                    f"| load={load_elapsed:.3f}s | save={save_elapsed:.3f}s "
                    f"| superpoints={num_sp}"
                )

            # cleanup
            del data, raw_to_voxel, raw_to_sp

    if total_scenes > 0 and total_superpoints > 0:
        avg_sp_time_ms = (total_compute_time / total_superpoints) * 1000.0
        avg_scene_time = total_compute_time / total_scenes
        print(
            f"Average compute time per scene: {avg_scene_time:.3f}s "
            f"({total_scenes} scenes)"
        )
        print(
            f"Average compute time per superpoint: {avg_sp_time_ms:.3f} ms "
            f"({total_superpoints} superpoints)"
        )
        print(
            f"Totals | load={total_load_time:.3f}s | compute={total_compute_time:.3f}s | save={total_save_time:.3f}s"
        )
    elif total_scenes > 0:
        print(f"Processed {total_scenes} scenes but found 0 superpoints.")

    print("Done.")
    try:
        log_file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
