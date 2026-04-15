import os.path as osp
import torch
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.data import Data, InstanceData
from src.utils.scannet import read_one_scan, read_one_test_scan
from src.datasets.scannet_config import (
    NYU40_2_SCANNET,
    SCANNET200_RAW_IDS,
    SCANNET200_NUM_CLASSES,
)

__all__ = ["read_scannet_scan", "resolve_label_map_file"]


def resolve_label_map_file(raw_dir: str) -> str:
    candidates = [
        osp.join(raw_dir, "scannetv2-labels.combined.tsv"),
        osp.join(raw_dir, "meta_data", "scannetv2-labels.combined.tsv"),
    ]
    for path in candidates:
        if osp.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find scannetv2-labels.combined.tsv in raw root. "
        "Tried: " + ", ".join(candidates)
    )


def read_scannet_scan(
        scan_dir: str,
        xyz: bool = True,
        rgb: bool = True,
        normal: bool = True,
        semantic: bool = True,
        instance: bool = True,
        remap: bool = True,
        label_type: str = "scannet200",
        label_map_file: str = None,
        axis_align: bool = False,
) -> Data:
    """Read a ScanNet scan.

    Expects the data to be saved under:
      raw/scans/sceneXXXX_YY/sceneXXXX_YY_vh_clean_2.ply (+ labels/json)
    """
    scan_dir = scan_dir[:-1] if scan_dir[-1] == '/' else scan_dir

    scan_name = osp.basename(scan_dir)
    stage_dir = osp.dirname(scan_dir)
    stage_dirname = osp.basename(stage_dir)
    raw_dir = osp.dirname(stage_dir)

    if label_map_file is None:
        label_map_file = resolve_label_map_file(raw_dir)

    if stage_dirname not in ['scans', 'scans_test']:
        raise ValueError(
            "Expected raw_dir/{scans, scans_test}/scan_name structure, "
            f"but parent directory is {stage_dirname}")

    if stage_dirname == 'scans':
        if label_type == "scannet20":
            label_to = "nyu40id"
        elif label_type == "scannet200":
            label_to = "id"
        else:
            raise ValueError(
                f"Unsupported label_type '{label_type}'. "
                "Expected 'scannet20' or 'scannet200'.")

        pos, color, n, y, obj = read_one_scan(
            stage_dir, scan_name, label_map_file, label_to=label_to,
            axis_align=axis_align)

        if remap:
            if label_type == "scannet20":
                y = torch.from_numpy(NYU40_2_SCANNET)[y]
            elif label_type == "scannet200":
                max_id = max(int(y.max()), max(SCANNET200_RAW_IDS))
                mapping = torch.full(
                    (max_id + 1,), SCANNET200_NUM_CLASSES, dtype=torch.long)
                for idx, raw_id in enumerate(SCANNET200_RAW_IDS):
                    if raw_id <= max_id:
                        mapping[raw_id] = idx
                y = mapping[y]
        pos_offset = torch.zeros_like(pos[0])
        data = Data(pos=pos, pos_offset=pos_offset, rgb=color, normal=n, y=y)
        idx = torch.arange(data.num_points)
        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)
        data.obj = InstanceData(idx, obj, count, y, dense=True)
    else:
        pos, color, n = read_one_test_scan(stage_dir, scan_name)
        pos_offset = torch.zeros_like(pos[0])
        data = Data(pos=pos, pos_offset=pos_offset, rgb=color, normal=n)

    # sanitize normals if needed
    if n is not None:
        idx = torch.where(n.norm(dim=1) == 0)[0]
        if idx.numel() > 0:
            n[idx] = torch.tensor([0, 0, 1], dtype=torch.float)

    if not xyz:
        data.pos = None
    if not rgb:
        data.rgb = None
    if not normal:
        data.normal = None
    if not semantic:
        data.y = None
    if not instance:
        data.obj = None

    return data
