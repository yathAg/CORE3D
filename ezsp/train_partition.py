#!/usr/bin/env python
"""Train EZ-SP partition CNN (ScanNet200 only), with cached preprocessing."""

import argparse
import datetime
import hashlib
import json
import os
import os.path as osp
import sys
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.storage import recursive_apply

EZSP_ROOT = osp.dirname(osp.abspath(__file__))
if EZSP_ROOT not in sys.path:
    sys.path.insert(0, EZSP_ROOT)

from src.datasets.scannet import read_scannet_scan, resolve_label_map_file
from src.datasets.scannet_config import SCANS, SCANNET200_NUM_CLASSES
from src.data import Data
from src.transforms.device import DataTo
from src.transforms.sampling import SaveNodeIndex, GridSampling3D
from src.transforms.neighbors import KNN
from src.transforms.graph import AdjacencyGraph
from src.transforms.point import PointFeatures, GroundElevation
from src.transforms.partition import GreedyContourPriorPartition
from src.nn import GraphNorm
from src.nn.stage import PointStage
from src.loss.partition_criterion import PartitionCriterion
from src.utils.output_partition import PartitionOutput
from src.utils import string_to_dtype
from src.metrics.semantic import ConfusionMatrix


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


def _parse_dtype(value: Optional[str]) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    value = value.strip().lower()
    if value in ("none", "null", "off"):
        return None
    return string_to_dtype(value)


def _dtype_to_str(value: Optional[torch.dtype]) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("torch.", "")


def _cast_cache_precision(
    data: Data,
    fp_dtype: Optional[torch.dtype],
    pos_dtype: Optional[torch.dtype],
) -> Data:
    if fp_dtype is None and pos_dtype is None:
        return data

    if fp_dtype is not None:
        def cast_fp(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                return x.to(fp_dtype)
            return x
        for key in data.keys:
            data[key] = recursive_apply(data[key], cast_fp)

    if pos_dtype is not None:
        if getattr(data, "pos", None) is not None and torch.is_tensor(data.pos):
            if data.pos.is_floating_point():
                data.pos = data.pos.to(pos_dtype)
        if getattr(data, "pos_offset", None) is not None and torch.is_tensor(data.pos_offset):
            if data.pos_offset.is_floating_point():
                data.pos_offset = data.pos_offset.to(pos_dtype)

    return data


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


def _init_run_dir(runs_root: str, run_dir: Optional[str]) -> str:
    final_dir = run_dir or osp.join(runs_root, _now_timestamp())
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def scene_id_from_rel(rel_path: str) -> str:
    return osp.basename(rel_path)


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


def preprocess_scene(
    raw_root: str,
    rel_path: str,
    voxel: float,
    quantize_coords: bool,
    partition_hf: List[str],
    num_classes: int,
    cache_path: str,
    preprocess_device: str,
    label_map_file: Optional[str],
    axis_align: bool,
    ground_model: Optional[str],
    ground_threshold: float,
    ground_xy_grid: float,
    ground_scale: float,
    knn_k: int,
    knn_r: float,
    cache_fp_dtype: Optional[torch.dtype],
    cache_pos_dtype: Optional[torch.dtype],
    overwrite: bool,
):
    if osp.exists(cache_path) and not overwrite:
        return

    raw_path = resolve_raw_path(raw_root, rel_path)
    if not osp.exists(raw_path):
        raise FileNotFoundError(f"Missing raw path: {raw_path}")

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
        axis_align=axis_align,
    )

    data = SaveNodeIndex(key='sub')(data)

    grid = GridSampling3D(
        size=voxel,
        quantize_coords=quantize_coords,
        hist_key='y',
        hist_size=num_classes + 1,
    )
    data = grid(data)

    if partition_hf:
        # If geometric features are requested, we need neighbors
        if needs_knn_features(partition_hf):
            if preprocess_device == "cpu":
                raise RuntimeError(
                    "Neighbor-based features require KNN on GPU (FRNN). "
                    "Set --preprocess_device cuda or drop geometric features."
                )
            device = torch.device(preprocess_device)
            data = DataTo(device)(data)
            data = KNN(k=knn_k, r_max=knn_r, verbose=False,
                       self_is_neighbor=False, save_as_csr=False)(data)
            data = data.to('cpu')

        data = PointFeatures(keys=partition_hf, overwrite=False)(data)

    # Optional ground elevation (only if requested)
    if ground_model is not None and partition_hf and 'elevation' in partition_hf:
        data = GroundElevation(
            model=ground_model,
            z_threshold=ground_threshold,
            xy_grid=ground_xy_grid,
            scale=ground_scale,
        )(data)

    # Drop neighbors to keep cache compact
    if getattr(data, 'neighbor_index', None) is not None:
        data.neighbor_index = None
    if getattr(data, 'neighbor_distance', None) is not None:
        data.neighbor_distance = None

    data = _cast_cache_precision(
        data,
        fp_dtype=cache_fp_dtype,
        pos_dtype=cache_pos_dtype,
    )
    data = data.to('cpu')
    data.scene_id = scene_id_from_rel(rel_path)

    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)


class CachedScanNet200Dataset(Dataset):
    def __init__(
        self,
        split: str,
        raw_root: str,
        cache_root: str,
        voxel: float,
        quantize_coords: bool,
        partition_hf: List[str],
        num_classes: int,
        preprocess_device: str,
        label_map_file: Optional[str],
        axis_align: bool,
        ground_model: Optional[str],
        ground_threshold: float,
        ground_xy_grid: float,
        ground_scale: float,
        knn_k: int,
        knn_r: float,
        cache_fp_dtype: Optional[torch.dtype],
        cache_pos_dtype: Optional[torch.dtype],
        overwrite_cache: bool = False,
        precompute: bool = True,
        limit: Optional[int] = None,
        drop_sub: bool = True,
    ):
        if split not in SCANS:
            raise ValueError(f"Unknown split '{split}', expected one of {list(SCANS.keys())}")
        self.split = split
        self.raw_root = raw_root
        self.cache_root = cache_root
        self.voxel = voxel
        self.quantize_coords = quantize_coords
        self.partition_hf = partition_hf
        self.num_classes = num_classes
        self.preprocess_device = preprocess_device
        self.label_map_file = label_map_file
        self.axis_align = axis_align
        self.ground_model = ground_model
        self.ground_threshold = ground_threshold
        self.ground_xy_grid = ground_xy_grid
        self.ground_scale = ground_scale
        self.knn_k = knn_k
        self.knn_r = knn_r
        self.cache_fp_dtype = cache_fp_dtype
        self.cache_pos_dtype = cache_pos_dtype
        self.overwrite_cache = overwrite_cache
        self.drop_sub = drop_sub
        self.scenes = SCANS[split][:limit] if limit is not None else SCANS[split]

        if precompute:
            for rel in self.scenes:
                scene_id = scene_id_from_rel(rel)
                cache_path = osp.join(self.cache_root, split, f"{scene_id}.pt")
                preprocess_scene(
                    raw_root=self.raw_root,
                    rel_path=rel,
                    voxel=self.voxel,
                    quantize_coords=self.quantize_coords,
                    partition_hf=self.partition_hf,
                    num_classes=self.num_classes,
                    cache_path=cache_path,
                    preprocess_device=self.preprocess_device,
                    label_map_file=self.label_map_file,
                    axis_align=self.axis_align,
                    ground_model=self.ground_model,
                    ground_threshold=self.ground_threshold,
                    ground_xy_grid=self.ground_xy_grid,
                    ground_scale=self.ground_scale,
                    knn_k=self.knn_k,
                    knn_r=self.knn_r,
                    cache_fp_dtype=self.cache_fp_dtype,
                    cache_pos_dtype=self.cache_pos_dtype,
                    overwrite=self.overwrite_cache,
                )

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        rel = self.scenes[idx]
        scene_id = scene_id_from_rel(rel)
        cache_path = osp.join(self.cache_root, self.split, f"{scene_id}.pt")
        if not osp.exists(cache_path):
            preprocess_scene(
                raw_root=self.raw_root,
                rel_path=rel,
                voxel=self.voxel,
                quantize_coords=self.quantize_coords,
                partition_hf=self.partition_hf,
                num_classes=self.num_classes,
                cache_path=cache_path,
                preprocess_device=self.preprocess_device,
                label_map_file=self.label_map_file,
                axis_align=self.axis_align,
                ground_model=self.ground_model,
                ground_threshold=self.ground_threshold,
                ground_xy_grid=self.ground_xy_grid,
                ground_scale=self.ground_scale,
                knn_k=self.knn_k,
                knn_r=self.knn_r,
                cache_fp_dtype=self.cache_fp_dtype,
                cache_pos_dtype=self.cache_pos_dtype,
                overwrite=self.overwrite_cache,
            )
        data = torch.load(cache_path)
        if self.drop_sub and getattr(data, "sub", None) is not None:
            data.sub = None
        return data


def save_checkpoint(first_stage, optimizer, epoch, out_dir, filename=None, metadata=None):
    os.makedirs(out_dir, exist_ok=True)
    state_dict = {f"net.first_stage.{k}": v for k, v in first_stage.state_dict().items()}
    ckpt = {
        "state_dict": state_dict,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "metadata": metadata or {"label_type": "scannet200"},
    }
    if filename is None:
        filename = f"epoch_{epoch:03d}.ckpt"
    out_path = osp.join(out_dir, filename)
    torch.save(ckpt, out_path)
    return out_path


def load_checkpoint(first_stage, optimizer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Load model weights
    model_dict = first_stage.state_dict()
    ckpt_dict = {}
    for k, v in ckpt.get("state_dict", {}).items():
        if k.startswith("net.first_stage."):
            ckpt_dict[k.replace("net.first_stage.", "")] = v
    model_dict.update({k: v for k, v in ckpt_dict.items() if k in model_dict})
    first_stage.load_state_dict(model_dict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch


def main():
    parser = argparse.ArgumentParser(description="Train EZ-SP partition CNN (ScanNet200 only)")
    parser.add_argument("--raw_root", required=True,
                        help="Path to ScanNet raw root (contains scans/ and scans_test/)")
    parser.add_argument("--label_map_file", default=None,
                        help="Optional path to scannetv2-labels.combined.tsv (auto-resolved if omitted)")
    parser.add_argument("--axis_align", dest="axis_align", action="store_true", default=False,
                        help="Apply axis alignment to raw ScanNet points")
    parser.add_argument("--no_axis_align", dest="axis_align", action="store_false",
                        help="Skip axis alignment (use raw ScanNet coordinates; default)")
    parser.add_argument("--cache_root", default=osp.join(EZSP_ROOT, "cache", "scannet200_partition"),
                        help="Cache directory for preprocessed data (inside ezsp)")
    parser.add_argument("--cache_auto", dest="cache_auto", action="store_true", default=True,
                        help="Append a hash of preprocessing params to cache_root (default)")
    parser.add_argument("--cache_legacy", dest="cache_auto", action="store_false",
                        help="Use cache_root as-is (no param hash)")
    parser.add_argument("--cache_fp_dtype", default="float16",
                        help="Floating-point dtype for cached tensors (e.g., float16, float32, none)")
    parser.add_argument("--cache_pos_dtype", default="float32",
                        help="Floating-point dtype for cached positions (e.g., float32, float16, none)")
    parser.add_argument("--ckpt_dir", default=osp.join(EZSP_ROOT, "checkpoints", "scannet200_partition"),
                        help="Checkpoint output directory (inside ezsp)")
    parser.add_argument("--runs_root", default=osp.join(EZSP_ROOT, "runs", "train_partition"),
                        help="Root directory for per-run logs/configs")
    parser.add_argument("--run_dir", default=None,
                        help="Optional explicit run directory (overrides runs_root)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=True,
                        help="Enable DataLoader pin_memory (default: True)")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false",
                        help="Disable DataLoader pin_memory")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=True,
                        help="Keep DataLoader workers alive (default: True)")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false",
                        help="Disable persistent DataLoader workers")
    parser.add_argument("--non_blocking", dest="non_blocking", action="store_true", default=True,
                        help="Use non_blocking transfers to GPU (default: True)")
    parser.add_argument("--blocking", dest="non_blocking", action="store_false",
                        help="Disable non_blocking transfers")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess_device", default=None,
                        help="Device for preprocessing (default: device if GPU else cpu)")

    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--quantize_coords", action="store_true", default=True)
    parser.add_argument("--no_quantize_coords", action="store_false", dest="quantize_coords")

    parser.add_argument("--partition_hf", default="rgb",
                        help="Comma-separated point features for CNN input")
    parser.add_argument("--cnn_dim_without_in", default="32,32,32")
    parser.add_argument("--cnn_kernel_size", type=int, default=3)
    parser.add_argument("--cnn_dilation", type=int, default=1)

    parser.add_argument("--contour_reg", default="0.02")
    parser.add_argument("--contour_min_size", default="5")
    parser.add_argument("--contour_edge_mode", default="unit")
    parser.add_argument("--contour_edge_reduce", default="add")
    parser.add_argument("--contour_k_isolated", type=int, default=0)
    parser.add_argument("--contour_sharding", type=float, default=None)
    parser.add_argument("--contour_max_iterations", type=int, default=-1)
    parser.add_argument("--contour_merge_only_small", action="store_true", default=False)

    parser.add_argument("--contour_knn", type=int, default=8)
    parser.add_argument("--contour_knn_r", type=float, default=2.0)

    parser.add_argument("--affinity_temperature", type=float, default=1.0)
    parser.add_argument("--adaptive_sampling_ratio", type=float, default=0.9)

    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--no_precompute", action="store_true", default=False)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)

    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save a checkpoint every N epochs (0 disables; default: 0)")
    parser.add_argument("--save_best_only", dest="save_best_only", action="store_true", default=True,
                        help="Save only the best epoch checkpoint (default: True)")
    parser.add_argument("--save_all", dest="save_best_only", action="store_false",
                        help="Save all checkpoints (no best-only filtering)")
    parser.add_argument("--save_last", dest="save_last", action="store_true", default=True,
                        help="Save/update last.ckpt each epoch (default: True)")
    parser.add_argument("--no_save_last", dest="save_last", action="store_false",
                        help="Disable last.ckpt saving")
    parser.add_argument("--val_every", type=int, default=10,
                        help="Run validation every N epochs (default: 10)")

    # Ground elevation parameters (optional)
    parser.add_argument("--ground_model", default=None,
                        help="Ground model for elevation (e.g., ransac). If None, skip.")
    parser.add_argument("--ground_threshold", type=float, default=0.05)
    parser.add_argument("--ground_xy_grid", type=float, default=0.2)
    parser.add_argument("--ground_scale", type=float, default=1.0)

    args = parser.parse_args()

    run_dir = _init_run_dir(args.runs_root, args.run_dir)
    log_path = osp.join(run_dir, "run.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[run] dir={run_dir}")

    device = torch.device(args.device)
    preprocess_device = args.preprocess_device
    if preprocess_device is None:
        preprocess_device = args.device if torch.cuda.is_available() else "cpu"

    partition_hf = _parse_list(args.partition_hf, str)
    cnn_dim_without_in = _parse_list(args.cnn_dim_without_in, int)
    contour_min_size = _parse_list(args.contour_min_size, int)
    contour_reg = _parse_reg(args.contour_reg)
    cache_fp_dtype = _parse_dtype(args.cache_fp_dtype)
    cache_pos_dtype = _parse_dtype(args.cache_pos_dtype)

    args.cache_fp_dtype = _dtype_to_str(cache_fp_dtype)
    args.cache_pos_dtype = _dtype_to_str(cache_pos_dtype)

    if args.num_workers == 0:
        args.persistent_workers = False

    if contour_min_size is None:
        contour_min_size = [5]
    if isinstance(contour_min_size, list) and len(contour_min_size) > 1:
        contour_min_size = [contour_min_size[0]]
    if contour_reg is None:
        contour_reg = 0.02
    if isinstance(contour_reg, list):
        contour_reg = contour_reg[0]

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
        "cache_fp_dtype": args.cache_fp_dtype,
        "cache_pos_dtype": args.cache_pos_dtype,
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

    num_classes = SCANNET200_NUM_CLASSES

    # Build datasets and cache
    train_ds = CachedScanNet200Dataset(
        split="train",
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        voxel=args.voxel,
        quantize_coords=args.quantize_coords,
        partition_hf=partition_hf,
        num_classes=num_classes,
        preprocess_device=preprocess_device,
        label_map_file=label_map_file,
        axis_align=args.axis_align,
        ground_model=args.ground_model,
        ground_threshold=args.ground_threshold,
        ground_xy_grid=args.ground_xy_grid,
        ground_scale=args.ground_scale,
        knn_k=args.contour_knn,
        knn_r=args.contour_knn_r,
        cache_fp_dtype=cache_fp_dtype,
        cache_pos_dtype=cache_pos_dtype,
        overwrite_cache=args.overwrite_cache,
        precompute=not args.no_precompute,
        limit=args.limit_train,
        drop_sub=True,
    )
    val_ds = CachedScanNet200Dataset(
        split="val",
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        voxel=args.voxel,
        quantize_coords=args.quantize_coords,
        partition_hf=partition_hf,
        num_classes=num_classes,
        preprocess_device=preprocess_device,
        label_map_file=label_map_file,
        axis_align=args.axis_align,
        ground_model=args.ground_model,
        ground_threshold=args.ground_threshold,
        ground_xy_grid=args.ground_xy_grid,
        ground_scale=args.ground_scale,
        knn_k=args.contour_knn,
        knn_r=args.contour_knn_r,
        cache_fp_dtype=cache_fp_dtype,
        cache_pos_dtype=cache_pos_dtype,
        overwrite_cache=args.overwrite_cache,
        precompute=not args.no_precompute,
        limit=args.limit_val,
        drop_sub=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # Build model
    num_hf_partition = sum(FEAT_SIZE[k] for k in partition_hf)

    first_stage = build_first_stage(
        num_hf_partition=num_hf_partition,
        cnn_dim_without_in_dim=cnn_dim_without_in,
        kernel_size=args.cnn_kernel_size,
        dilation=args.cnn_dilation,
    ).to(device)

    criterion = PartitionCriterion(
        affinity_temperature=args.affinity_temperature,
        adaptive_sampling_ratio=args.adaptive_sampling_ratio,
        num_classes=num_classes,
        sharding=args.contour_sharding,
    )

    optimizer = torch.optim.Adam(first_stage.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(first_stage, optimizer, args.resume, device)

    best_miou = -float("inf")
    best_ckpt_path = None

    partitioner = GreedyContourPriorPartition(
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

    for epoch in range(start_epoch, args.epochs + 1):
        first_stage.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            if isinstance(batch, Data):
                data = batch
            else:
                data = batch
            data = data.to(device, non_blocking=args.non_blocking)

            # Build graph
            data = KNN(k=args.contour_knn, r_max=args.contour_knn_r,
                       verbose=False, self_is_neighbor=False, save_as_csr=False)(data)
            data = AdjacencyGraph(k=args.contour_knn, w=-1)(data)

            # Build input features
            data.add_keys_to(keys=partition_hf, to='x', delete_after=False)

            out = first_stage(
                data.x,
                data.norm_index(mode='graph'),
                pos=data.pos,
                diameter=None,
                node_size=getattr(data, 'node_size', None),
                super_index=data.super_index,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                coords=getattr(data, 'coords', None),
                batch=getattr(data, 'batch', None),
                x_mlp=getattr(data, 'x_mlp', None),
            )
            x = out[0] if isinstance(out, tuple) else out

            output = PartitionOutput(y=data.y, x=x, edge_index=data.edge_index)
            loss, _ = criterion(output)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)

        # Validation (optional)
        do_val = args.val_every > 0 and (epoch % args.val_every == 0 or epoch == args.epochs)
        if do_val:
            first_stage.eval()
            val_loss = 0.0
            cm = ConfusionMatrix(num_classes).to(device)
            with torch.no_grad():
                for batch in val_loader:
                    data = batch.to(device, non_blocking=args.non_blocking)
                    data = KNN(k=args.contour_knn, r_max=args.contour_knn_r,
                               verbose=False, self_is_neighbor=False, save_as_csr=False)(data)
                    data = AdjacencyGraph(k=args.contour_knn, w=-1)(data)

                    data.add_keys_to(keys=partition_hf, to='x', delete_after=False)
                    out = first_stage(
                        data.x,
                        data.norm_index(mode='graph'),
                        pos=data.pos,
                        diameter=None,
                        node_size=getattr(data, 'node_size', None),
                        super_index=data.super_index,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        coords=getattr(data, 'coords', None),
                        batch=getattr(data, 'batch', None),
                        x_mlp=getattr(data, 'x_mlp', None),
                    )
                    x = out[0] if isinstance(out, tuple) else out

                    output = PartitionOutput(y=data.y, x=x, edge_index=data.edge_index)
                    loss, _ = criterion(output)
                    val_loss += loss.item()

                    # Partition for purity metrics
                    data_part = data.clone()
                    data_part.x = x
                    nag = partitioner(data_part)
                    sp = nag[1]
                    y_oracle = sp.y[:, :num_classes].argmax(dim=1)
                    cm(y_oracle, sp.y)

            val_loss /= max(len(val_loader), 1)
            miou = float(cm.miou())
            oa = float(cm.oa())
            macc = float(cm.macc())
        else:
            val_loss = None
            miou = None
            oa = None
            macc = None

        elapsed = time.time() - t0
        if do_val:
            val_str = f"val_loss={val_loss:.4f} | oMiou={miou:.2f} oOA={oa:.2f} oMAcc={macc:.2f}"
        else:
            val_str = "val_loss=NA | oMiou=NA oOA=NA oMAcc=NA"
        print(
            f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f} "
            f"{val_str} | {elapsed:.1f}s"
        )

        ckpt_meta = {
            "label_type": "scannet200",
            "voxel": args.voxel,
            "quantize_coords": args.quantize_coords,
            "partition_hf": partition_hf,
            "cnn_dim_without_in": cnn_dim_without_in,
            "cnn_kernel_size": args.cnn_kernel_size,
            "cnn_dilation": args.cnn_dilation,
            "axis_align": args.axis_align,
            "label_map_file": label_map_file,
            "ground_model": args.ground_model,
            "ground_threshold": args.ground_threshold,
            "ground_xy_grid": args.ground_xy_grid,
            "ground_scale": args.ground_scale,
            "contour_knn": args.contour_knn,
            "contour_knn_r": args.contour_knn_r,
        }

        if args.save_last:
            save_checkpoint(first_stage, optimizer, epoch, args.ckpt_dir,
                            filename="last.ckpt", metadata=ckpt_meta)

        if do_val and miou is not None:
            if miou > best_miou:
                best_miou = miou
                new_best = save_checkpoint(
                    first_stage, optimizer, epoch, args.ckpt_dir, metadata=ckpt_meta)
                if args.save_best_only and best_ckpt_path and best_ckpt_path != new_best:
                    try:
                        os.remove(best_ckpt_path)
                    except FileNotFoundError:
                        pass
                best_ckpt_path = new_best

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(first_stage, optimizer, epoch, args.ckpt_dir, metadata=ckpt_meta)

    print("Training complete.")
    try:
        log_file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
