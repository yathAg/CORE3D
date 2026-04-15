#!/usr/bin/env python3
"""Feature and geometry helpers for scene graph construction."""

from typing import List, Optional
import colorsys

import numpy as np


PLANE_AXES = {
    'xy': (0, 1),
    'xz': (0, 2),
    'yz': (1, 2)
}


def apply_axis_align(xyz: np.ndarray, axis_align: Optional[np.ndarray]) -> np.ndarray:
    if axis_align is None or axis_align.shape != (4, 4):
        return xyz
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    aligned = xyz_h @ axis_align.T
    return aligned[:, :3]


def majority_label_from_values(labels: np.ndarray) -> Optional[int]:
    if labels.size == 0:
        return None
    if labels.min() < 0:
        labels = labels[labels >= 0]
        if labels.size == 0:
            return None
    labels = labels.astype(np.int64, copy=False)
    counts = np.bincount(labels)
    max_count = counts.max()
    cand = np.flatnonzero(counts == max_count)
    if cand.size == 1:
        return int(cand[0])
    cand_set = set(cand.tolist())
    for val in labels:
        if val in cand_set:
            return int(val)
    return int(cand[0])


def majority_label(sem_mask: np.ndarray, mask: np.ndarray) -> Optional[int]:
    pts = sem_mask[mask]
    return majority_label_from_values(pts)


def majority_color_from_values(colors_int: np.ndarray) -> Optional[List[int]]:
    if colors_int.size == 0:
        return None
    if colors_int.dtype != np.int32:
        colors_int = colors_int.astype(np.int32, copy=False)
    colors_int = np.clip(colors_int, 0, 255)
    maj = []
    for c in range(colors_int.shape[1]):
        counts = np.bincount(colors_int[:, c], minlength=256)
        maj.append(int(counts.argmax()))
    return maj


def majority_color(colors: np.ndarray, mask: np.ndarray) -> Optional[List[int]]:
    pts = colors[mask]
    if pts.size == 0:
        return None
    pts_int = np.clip(np.rint(pts), 0, 255).astype(np.int32)
    return majority_color_from_values(pts_int)


def iqr_mean_color_from_values(colors_int: np.ndarray) -> Optional[List[int]]:
    if colors_int.size == 0:
        return None
    if colors_int.dtype != np.int32:
        colors_int = colors_int.astype(np.int32, copy=False)
    colors_int = np.clip(colors_int, 0, 255)
    q1 = np.percentile(colors_int, 25, axis=0)
    q3 = np.percentile(colors_int, 75, axis=0)
    keep = (colors_int >= q1) & (colors_int <= q3)
    keep_mask = keep.all(axis=1)
    subset = colors_int[keep_mask]
    if subset.size == 0:
        subset = colors_int
    mean = subset.mean(axis=0)
    return np.clip(np.rint(mean), 0, 255).astype(np.int32).tolist()


def iqr_mean_color(colors: np.ndarray, mask: np.ndarray) -> Optional[List[int]]:
    pts = colors[mask]
    if pts.size == 0:
        return None
    pts_int = np.clip(np.rint(pts), 0, 255).astype(np.int32)
    return iqr_mean_color_from_values(pts_int)


def name_basic11(rgb: List[int]) -> str:
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360.0
    maxc, minc = max(r, g, b), min(r, g, b)
    delta = maxc - minc

    if delta < 0.02 or s < 0.08:
        if v < 0.15:
            return 'black'
        if v > 0.85:
            return 'white'
        return 'gray'

    if h_deg < 20 or h_deg >= 340:
        base = 'red'
    elif h_deg < 45:
        base = 'orange'
    elif h_deg < 65:
        base = 'yellow'
    elif h_deg < 170:
        base = 'green'
    elif h_deg < 250:
        base = 'blue'
    elif h_deg < 290:
        base = 'purple'
    elif h_deg < 330:
        base = 'pink'
    else:
        base = 'red'

    if base in {'red', 'orange', 'yellow'} and 0.15 < v < 0.6 and s < 0.7:
        base = 'brown'

    if v > 0.75:
        adjective = 'light '
    elif v < 0.35:
        adjective = 'dark '
    else:
        adjective = ''

    return f'{adjective}{base}'.strip()


def centroid_from_mask(xyz: np.ndarray, mask: np.ndarray) -> Optional[List[float]]:
    pts = xyz[mask]
    return centroid_from_points(pts)


def centroid_from_points(pts: np.ndarray) -> Optional[List[float]]:
    if pts.size == 0:
        return None
    return pts.mean(axis=0).astype(np.float32).tolist()


def direction_label(delta: np.ndarray, plane: str) -> str:
    if plane not in PLANE_AXES:
        raise ValueError(f'Unsupported direction plane: {plane}')
    h_idx, v_idx = PLANE_AXES[plane]
    h = delta[h_idx]
    v = delta[v_idx]
    if abs(h) >= abs(v):
        return 'right' if h >= 0 else 'left'
    return 'up' if v >= 0 else 'down'


def rotation_from_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def direction_label_observer(delta: np.ndarray, obs_rot: Optional[np.ndarray]) -> str:
    if obs_rot is not None:
        delta = obs_rot.T @ delta
    ax = np.abs(delta)
    idx = int(ax.argmax())
    if idx == 0:
        return 'right' if delta[0] >= 0 else 'left'
    if idx == 1:
        return 'front' if delta[1] >= 0 else 'back'
    return 'up' if delta[2] >= 0 else 'down'
