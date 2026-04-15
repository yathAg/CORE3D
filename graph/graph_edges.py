#!/usr/bin/env python3
"""Relationship and edge feature helpers."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from graph.graph_features import direction_label, direction_label_observer


REL_NAMES = ['near', 'left', 'right', 'front', 'back', 'above', 'below']
REL_TO_ID = {name: idx for idx, name in enumerate(REL_NAMES)}


def build_relationships(node_ids: List[int],
                        centroids: Dict[int, np.ndarray],
                        knn_k: Optional[int],
                        direction_plane: str,
                        use_cardinals: bool,
                        observer_rot: Optional[np.ndarray]) -> Tuple[List[List], Dict[int, List[int]], Dict[str, Dict]]:
    relationships: List[List] = []
    neighbors: Dict[int, List[int]] = {}
    edge_features: Dict[str, Dict] = {}

    for nid in node_ids:
        candidates = [oid for oid in node_ids if oid != nid]
        if knn_k is not None and knn_k > 0:
            candidates.sort(key=lambda x: (float(np.linalg.norm(centroids[x] - centroids[nid])), x))
            selected = candidates[:knn_k]
        else:
            selected = candidates
        neighbors[nid] = selected
        for tgt in selected:
            delta = centroids[tgt] - centroids[nid]
            dist = float(np.linalg.norm(delta))
            if observer_rot is not None:
                dir_raw = direction_label_observer(delta, observer_rot)
            else:
                dir_raw = direction_label(delta, direction_plane)
            if use_cardinals:
                if dir_raw == 'up':
                    rel_name = 'above'
                elif dir_raw == 'down':
                    rel_name = 'below'
                else:
                    rel_name = dir_raw
            else:
                rel_name = 'near'
            relationships.append([int(nid), int(tgt), REL_TO_ID[rel_name], rel_name])
            edge_features[f'{nid}->{tgt}'] = {'distance': dist}

    return relationships, neighbors, edge_features
