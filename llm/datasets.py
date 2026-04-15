#!/usr/bin/env python3
"""Dataset loading helpers for LLM benchmarks."""

import json
import os.path as osp
from typing import Any, Dict, List


def load_dataset(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    root = cfg["data"]["root"]
    split_entry = cfg["data"]["splits"].get(split, cfg["data"]["splits"].get("all"))
    paths = split_entry if isinstance(split_entry, list) else [split_entry]
    rows: List[Dict[str, Any]] = []
    for p in paths:
        path = p if osp.isabs(p) else osp.join(root, p)
        if not osp.exists(path):
            continue
        with open(path) as f:
            rows.extend(json.load(f))
    return rows


def _load_sqa3d_from_paths(root: str, split_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    q_path = split_cfg.get("questions")
    a_path = split_cfg.get("annotations")
    if not q_path or not a_path:
        return []
    q_path = q_path if osp.isabs(q_path) else osp.join(root, q_path)
    a_path = a_path if osp.isabs(a_path) else osp.join(root, a_path)
    if not osp.exists(q_path) or not osp.exists(a_path):
        return []
    with open(q_path, "r", encoding="utf-8") as f:
        q_data = json.load(f).get("questions", [])
    with open(a_path, "r", encoding="utf-8") as f:
        a_data = json.load(f).get("annotations", [])
    question_map: Dict[int, Dict[str, Any]] = {}
    for item in q_data:
        question_map[item["question_id"]] = {
            "scene_id": item.get("scene_id"),
            "situation": item.get("situation", ""),
            "alternative_situation": item.get("alternative_situation", []),
            "question": item.get("question", ""),
        }
    rows: List[Dict[str, Any]] = []
    for item in a_data:
        q = question_map.get(item.get("question_id"))
        if not q:
            continue
        answers = [a.get("answer", "") for a in item.get("answers", []) if a.get("answer") is not None]
        if not answers:
            answers = [""]
        rows.append(
            {
                "scene_id": item.get("scene_id", q.get("scene_id")),
                "question_id": item.get("question_id"),
                "question": q.get("question", ""),
                "situation": q.get("situation", ""),
                "alt_situations": q.get("alternative_situation", []),
                "answers": answers,
                "question_type": item.get("question_type"),
                "answer_type": item.get("answer_type"),
                "position": item.get("position"),
                "rotation": item.get("rotation"),
            }
        )
    return rows


def load_sqa3d_dataset(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    root = cfg["data"]["root"]
    split_entry = cfg["data"]["splits"].get(split, cfg["data"]["splits"].get("all"))
    if split_entry is None:
        return []
    if isinstance(split_entry, list):
        rows: List[Dict[str, Any]] = []
        for entry in split_entry:
            if isinstance(entry, str):
                rows.extend(load_sqa3d_dataset(cfg, entry))
            elif isinstance(entry, dict):
                rows.extend(_load_sqa3d_from_paths(root, entry))
        return rows
    if isinstance(split_entry, dict):
        return _load_sqa3d_from_paths(root, split_entry)
    return []
