#!/usr/bin/env python3
"""Benchmark runners and evaluation helpers for LLM inference."""

import csv
import json
import os
import os.path as osp
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import tqdm

from llm.core import (
    _build_script_args,
    _collect_vllm_engine_metrics,
    _init_stats_totals,
    _coerce_prediction_text,
    _shutdown_llm,
    _write_run_metrics_csv,
    call_model_batch_vllm,
)
from llm.datasets import load_dataset, load_sqa3d_dataset
from llm.prompts import format_prompt_caption, format_prompt_qa, format_prompt_selection

# Optional text metrics (used by Scan2Cap / ScanQA). Loaded lazily.
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
except Exception as _e:  # pragma: no cover
    PTBTokenizer = Bleu = Meteor = Rouge = Cider = None
    _PYCOCO_IMPORT_ERROR = _e
else:
    _PYCOCO_IMPORT_ERROR = None

# 200 ScanNet200 classes for Reason3D class-id mapping.
SCANNET200_CLASSES = [
    'wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom',
    'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle',
    'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher',
    'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
    'projector screen', 'divider', 'laundry detergent', 'bathroom counter',
    'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
    'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
    'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
    'coffee kettle', 'structure', 'shower head', 'keyboard piano',
    'case of water bottles', 'coat rack', 'storage organizer', 'folded chair',
    'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
    'luggage', 'mattress'
]
CLASS_NAME_TO_ID = {name.lower(): idx for idx, name in enumerate(SCANNET200_CLASSES)}


def _init_counter() -> Dict[str, int]:
    return {"true": 0, "false": 0, "evaluated": 0, "skipped": 0, "target_missing": 0, "target_present": 0}


def load_graph(graph_root: str, scene_id: str) -> Dict[str, Any]:
    with open(osp.join(graph_root, f"{scene_id}.json")) as f:
        return json.load(f)


def trim_objects(scene_graph: Dict[str, Any], max_objects: int, target_ids: List[Optional[int]]) -> List[int]:
    meta = scene_graph.get("object_metadata", {}) or {}
    ids = [int(i) for i in meta.keys()]

    def sort_key(i: int):
        m = meta.get(str(i)) or meta.get(i) or {}
        sz = m.get("size", 0) or 0
        return (-sz, i)

    ids.sort(key=sort_key)
    keep = set(ids[:max_objects])
    for tid in target_ids:
        if tid is not None:
            keep.add(tid)
    kept = [i for i in ids if i in keep]
    for tid in target_ids:
        if tid is not None and tid not in kept:
            kept.append(tid)
    return sorted(set(kept))


def _graph_object_ids(scene_graph: Dict[str, Any]) -> List[int]:
    ids = set()
    meta = scene_graph.get("object_metadata", {}) or {}
    for key in meta.keys():
        try:
            ids.add(int(key))
        except Exception:
            continue
    if not ids:
        fallback = scene_graph.get("objects", {}) or {}
        for key in fallback.keys():
            try:
                ids.add(int(key))
            except Exception:
                continue
    return sorted(ids)


def build_subset_map(scene_graph: Dict[str, Any]) -> Dict[int, str]:
    obj_meta = scene_graph.get("object_metadata", {}) or {}
    obj_labels: Dict[int, Optional[str]] = {}

    fallback_labels = scene_graph.get("objects", {}) or {}
    for raw_oid, meta in obj_meta.items():
        try:
            oid = int(raw_oid)
        except Exception:
            oid = raw_oid
        label = meta.get("label") if isinstance(meta, dict) else None
        if label is None:
            label = fallback_labels.get(str(raw_oid)) or fallback_labels.get(raw_oid)
        obj_labels[oid] = label

    for raw_oid, label in fallback_labels.items():
        try:
            oid = int(raw_oid)
        except Exception:
            oid = raw_oid
        if oid not in obj_labels:
            obj_labels[oid] = label

    counts = Counter(lbl for lbl in obj_labels.values() if lbl is not None)
    subset_map: Dict[int, str] = {}
    for oid, label in obj_labels.items():
        if label is None:
            subset = "unknown"
        else:
            subset = "unique" if counts[label] == 1 else "multiple"
        subset_map[oid] = subset
    return subset_map


def enrich_class_ids(scene_graph: Dict[str, Any]):
    """Populate class_id/class_name from labels for Reason3D."""
    obj_meta = scene_graph.get("object_metadata", {}) or {}
    obj_names = scene_graph.get("objects", {}) or {}
    for oid_key in list(obj_meta.keys()):
        meta = obj_meta[oid_key] or {}
        label = meta.get("label") or obj_names.get(str(oid_key)) or obj_names.get(oid_key)
        class_id = None
        if isinstance(label, str):
            class_id = CLASS_NAME_TO_ID.get(label.lower())
        if class_id is not None:
            meta["class_id"] = class_id
            meta["class_name"] = SCANNET200_CLASSES[class_id]
        obj_meta[oid_key] = meta
    scene_graph["object_metadata"] = obj_meta


def _compute_text_metrics(preds: List[str], refs: List[List[str]]) -> Optional[Dict[str, float]]:
    if not preds:
        return None
    if PTBTokenizer is None:
        raise RuntimeError(
            "pycocoevalcap is not installed. Install with "
            "`pip install git+https://github.com/tylin/coco-caption#subdirectory=pycocoevalcap` "
            f"(import error: {_PYCOCO_IMPORT_ERROR})."
        )
    gts = {str(i): [{"caption": r} for r in ref_list] for i, ref_list in enumerate(refs)}
    res = {str(i): [{"caption": pred}] for i, pred in enumerate(preds)}
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    out: Dict[str, float] = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_tok, res_tok)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                out[m] = float(sc)
        else:
            out[method] = float(score)
    return {
        "Cider": out.get("CIDEr", 0.0) * 100.0,
        "Bleu": out.get("BLEU-4", 0.0) * 100.0,
        "Meteor": out.get("METEOR", 0.0) * 100.0,
        "RougeL": out.get("ROUGE_L", 0.0) * 100.0,
    }


def _clean_answer_sqa3d(data: str) -> str:
    data = data.lower()
    data = re.sub(r"[ ]+$", "", data)
    data = re.sub(r"^[ ]+", "", data)
    data = re.sub(r" {2,}", " ", data)
    data = re.sub(r"\.[ ]{2,}", ". ", data)
    data = re.sub(r"[^a-zA-Z0-9,\'\s\-:]+", "", data)
    data = re.sub(r"ç", "c", data)
    data = re.sub(r"’", "'", data)
    data = re.sub(r"\bletf\b", "left", data)
    data = re.sub(r"\blet\b", "left", data)
    data = re.sub(r"\btehre\b", "there", data)
    data = re.sub(r"\brigth\b", "right", data)
    data = re.sub(r"\brght\b", "right", data)
    data = re.sub(r"\bbehine\b", "behind", data)
    data = re.sub(r"\btv\b", "TV", data)
    data = re.sub(r"\bchai\b", "chair", data)
    data = re.sub(r"\bwasing\b", "washing", data)
    data = re.sub(r"\bwaslked\b", "walked", data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo\'[ ]+clock\b", "o'clock", data)
    data = re.sub(r"\b0\b", "zero", data)
    data = re.sub(r"\bnone\b", "zero", data)
    data = re.sub(r"\b1\b", "one", data)
    data = re.sub(r"\b2\b", "two", data)
    data = re.sub(r"\b3\b", "three", data)
    data = re.sub(r"\b4\b", "four", data)
    data = re.sub(r"\b5\b", "five", data)
    data = re.sub(r"\b6\b", "six", data)
    data = re.sub(r"\b7\b", "seven", data)
    data = re.sub(r"\b8\b", "eight", data)
    data = re.sub(r"\b9\b", "nine", data)
    data = re.sub(r"\b10\b", "ten", data)
    data = re.sub(r"\b11\b", "eleven", data)
    data = re.sub(r"\b12\b", "twelve", data)
    data = re.sub(r"\b13\b", "thirteen", data)
    data = re.sub(r"\b14\b", "fourteen", data)
    data = re.sub(r"\b15\b", "fifteen", data)
    data = re.sub(r"\b16\b", "sixteen", data)
    data = re.sub(r"\b17\b", "seventeen", data)
    data = re.sub(r"\b18\b", "eighteen", data)
    data = re.sub(r"\b19\b", "nineteen", data)
    data = re.sub(r"\b20\b", "twenty", data)
    data = re.sub(r"\b23\b", "twenty-three", data)
    data = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", data)
    data = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bbackwards\b", "backward", data)
    return data


def _answer_match_sqa3d(pred: str, gts: List[str]) -> Tuple[int, int]:
    if len(pred) == 0:
        return 0, 0
    if pred in gts:
        return 1, 1
    for gt in gts:
        if "".join(pred.split()) in "".join(gt.split()) or "".join(gt.split()) in "".join(pred.split()):
            return 0, 1
    return 0, 0


def _compute_sqa3d_em1(preds: List[str], refs: List[List[str]]) -> Optional[float]:
    if not preds:
        return None
    correct = 0
    total = 0
    for pred, ref_list in zip(preds, refs):
        pred = pred or ""
        pred = pred.strip()
        if len(pred) > 1:
            if pred.endswith("."):
                pred = pred[:-1]
            pred = pred[0].lower() + pred[1:]
        pred = _clean_answer_sqa3d(pred)
        cleaned_refs = [_clean_answer_sqa3d(r) for r in (ref_list or [])]
        em_flag, _ = _answer_match_sqa3d(pred, cleaned_refs)
        correct += em_flag
        total += 1
    return correct / total if total else None


def _normalize_id_list(pred: Any) -> List[int]:
    if pred is None:
        return []
    if isinstance(pred, (int, float, str)):
        try:
            return [int(pred)]
        except Exception:
            return []
    if isinstance(pred, dict) and "id" in pred:
        return _normalize_id_list(pred["id"])
    if isinstance(pred, list):
        out = []
        for p in pred:
            out.extend(_normalize_id_list(p))
        return out
    return []


# ------------------------------ Benchmark runners ------------------------------ #
def _run_scanrefer_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_dataset(cfg, split)
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    graph_run_args_path = osp.join(osp.dirname(osp.normpath(graph_root)), "run_args.json")
    graph_run_args = json.load(open(graph_run_args_path)) if osp.exists(graph_run_args_path) else None

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {
        "timestamp": run_ts,
        "script_args": script_args,
        "graph_root_resolved": graph_root,
        "graph_run_args_path": graph_run_args_path if osp.exists(graph_run_args_path) else None,
        "graph_run_args": graph_run_args,
        "save_per_query": args.save_per_query or bool(args.scene),
        "split": split,
    }
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    scene_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, q in enumerate(queries):
        scene_groups.setdefault(q["scene_id"], []).append((idx, q))

    per_query: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, float]] = {}
    subset_cache: Dict[str, Dict[int, str]] = {}
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    skipped_missing_graph = 0

    def ensure_scene(scene_id: str) -> Dict[str, float]:
        if scene_id not in stats:
            stats[scene_id] = {
                "true": 0,
                "false": 0,
                "evaluated": 0,
                "skipped": 0,
                "target_missing": 0,
                "target_present": 0,
                "with_graph": 0,
                "subsets": {
                    "unique": _init_counter(),
                    "multiple": _init_counter(),
                    "unknown": _init_counter(),
                },
            }
        return stats[scene_id]

    def update_stats(entry: Dict[str, Any], result: Dict[str, Any]):
        scene_id = entry["scene_id"]
        s = ensure_scene(scene_id)
        subset = entry.get("subset")
        nonlocal preds_written
        pred_obj_id = result.get("prediction")
        target_obj_id_int = entry["target_obj_id_int"]
        parse_fail = result.get("parse_fail", False)
        target_in_graph = entry.get("target_in_graph")

        pred_id_int = None
        if not parse_fail and pred_obj_id is not None:
            try:
                pred_id_int = int(pred_obj_id)
            except Exception:
                pred_id_int = None
        correct = (
            pred_id_int is not None
            and target_obj_id_int is not None
            and pred_id_int == int(target_obj_id_int)
        )

        s["evaluated"] += 1
        if target_in_graph is False or target_in_graph is None:
            s["target_missing"] += 1
        else:
            s["target_present"] += 1
        if correct:
            s["true"] += 1
        else:
            s["false"] += 1

        if subset in ("unique", "multiple"):
            cs = s["subsets"][subset]
            cs["evaluated"] += 1
            if target_in_graph is False or target_in_graph is None:
                cs["target_missing"] += 1
            else:
                cs["target_present"] += 1
            if correct:
                cs["true"] += 1
            else:
                cs["false"] += 1

        out_obj = {
            "scene": scene_id,
            "query": entry["query_text"],
            "prediction": pred_id_int,
            "gt_object_id": entry["target_obj_id"],
            "gt_object_id_1indexed": entry["target_obj_id_int"],
            "subset": subset,
            "correct": correct,
            "parse_fail": parse_fail,
            "target_in_graph": bool(target_in_graph),
        }
        if args.allow_rationale:
            out_obj["rationale"] = result.get("rationale", "")
        label_lookup = entry.get("obj_label_lookup") or {}
        size_lookup = entry.get("obj_size_lookup") or {}
        if pred_id_int is None:
            pred_label = None
            pred_size = None
        elif pred_id_int in label_lookup:
            pred_label = label_lookup.get(pred_id_int)
            pred_size = size_lookup.get(pred_id_int)
        else:
            pred_label = "<not_in_graph>"
            pred_size = None
        pred_row = {
            "scene_id": scene_id,
            "idx": entry["idx"],
            "gt_object_id": entry["target_obj_id"],
            "gt_object_id_1indexed": entry["target_obj_id_int"],
            "prediction": pred_id_int,
            "parse_fail": parse_fail,
            "skipped": False,
            "target_in_graph": bool(target_in_graph),
            "query_text": entry.get("query_text"),
            "subset": subset,
            "target_label": entry.get("target_label"),
            "target_size": entry.get("target_size"),
            "target_rank_by_size": entry.get("target_rank_by_size"),
            "num_same_class": entry.get("num_same_class"),
            "pred_label": pred_label,
            "pred_size": pred_size,
        }
        preds_f.write(json.dumps(pred_row) + "\n")
        preds_written += 1
        if preds_written % preds_flush_every == 0:
            preds_f.flush()
        if args.save_per_query or args.scene:
            per_query.append(out_obj)
            prefix = f"{output_prefix}_" if output_prefix else ""
            out_path = osp.join(out_root, f"{prefix}{scene_id}_q{entry['idx']}.json")
            with open(out_path, "w") as f:
                json.dump(out_obj, f, indent=2)

    def process_entries(entries: List[Dict[str, Any]]):
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            update_stats(entry, result)
        entries.clear()

    for scene_id in tqdm.tqdm(sorted(scene_groups.keys()), desc="scenes"):
        scene_queries = scene_groups[scene_id]
        s = ensure_scene(scene_id)

        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += len(scene_queries)
            s["skipped"] += len(scene_queries)
            continue

        s["with_graph"] += len(scene_queries)
        graph = load_graph(graph_root, scene_id)
        graph_ids = set(_graph_object_ids(graph))

        subset_map = subset_cache.get(scene_id) or build_subset_map(graph)
        subset_cache[scene_id] = subset_map

        # Per-scene lookups for failure-analysis logging. Reused across all
        # queries in this scene; passed by reference into each entry.
        _meta = graph.get("object_metadata", {}) or {}
        label_lookup: Dict[int, Optional[str]] = {}
        size_lookup: Dict[int, int] = {}
        for _k, _m in _meta.items():
            try:
                _i = int(_k)
            except Exception:
                continue
            label_lookup[_i] = (_m or {}).get("label")
            size_lookup[_i] = int((_m or {}).get("size", 0) or 0)
        label_count = Counter(v for v in label_lookup.values() if v is not None)
        ids_by_size = sorted(label_lookup.keys(), key=lambda i: -size_lookup.get(i, 0))
        rank_by_size = {oid: rank for rank, oid in enumerate(ids_by_size)}

        target_ids_all: List[int] = []
        for _, q in scene_queries:
            tgt = q.get("object_id")
            try:
                raw = int(tgt) if tgt is not None else None
            except Exception:
                raw = None
            target_ids_all.append(raw + cfg["data"]["id_offset"] if raw is not None else None)

        keep_ids = trim_objects(graph, args.max_objects, target_ids_all)

        entries: List[Dict[str, Any]] = []
        for orig_idx, q in scene_queries:
            target_obj_id = q.get("object_id")
            try:
                raw_target = int(target_obj_id) if target_obj_id is not None else None
            except Exception:
                raw_target = None
            target_obj_id_int = raw_target + cfg["data"]["id_offset"] if raw_target is not None else None
            query_text = q.get("description") or ""
            subset = subset_map.get(target_obj_id_int) or subset_map.get(str(target_obj_id_int)) or "unknown"
            target_in_graph = target_obj_id_int in graph_ids if target_obj_id_int is not None else False

            prompt = format_prompt_selection(
                scene_id,
                graph,
                query_text,
                cfg["prompt"]["instruction"],
                keep_ids,
                include_edges=args.use_edges,
                neighbor_cap=args.neighbor_cap,
                prompt_format=args.prompt_format,
                use_attributes=args.use_attributes,
                use_colors=args.use_colors,
                use_size=args.use_size,
            )

            target_label = label_lookup.get(target_obj_id_int) if target_obj_id_int is not None else None
            entries.append(
                {
                    "idx": orig_idx,
                    "scene_id": scene_id,
                    "prompt": prompt,
                    "query_text": query_text,
                    "keep_ids": keep_ids,
                    "target_obj_id": target_obj_id,
                    "target_obj_id_int": target_obj_id_int,
                    "subset": subset,
                    "target_in_graph": target_in_graph,
                    "target_label": target_label,
                    "target_size": size_lookup.get(target_obj_id_int) if target_obj_id_int is not None else None,
                    "target_rank_by_size": rank_by_size.get(target_obj_id_int) if target_obj_id_int is not None else None,
                    "num_same_class": label_count.get(target_label, 0) if target_label else 0,
                    "obj_label_lookup": label_lookup,
                    "obj_size_lookup": size_lookup,
                }
            )
            if len(entries) >= args.batch_size:
                process_entries(entries)
        process_entries(entries)

    preds_f.flush()
    preds_f.close()

    total_true = sum(v["true"] for v in stats.values())
    total_false = sum(v["false"] for v in stats.values())
    total_eval = sum(v["evaluated"] for v in stats.values())
    total_skipped = sum(v["skipped"] for v in stats.values())
    total_target_missing = sum(v["target_missing"] for v in stats.values())
    total_target_present = sum(v["target_present"] for v in stats.values())
    total_acc = total_true / total_eval if total_eval else None
    total_target_missing_rate = (total_target_missing / total_eval) if total_eval else None

    def subset_totals(key: str):
        t_true = sum(v["subsets"][key]["true"] for v in stats.values())
        t_false = sum(v["subsets"][key]["false"] for v in stats.values())
        t_eval = sum(v["subsets"][key]["evaluated"] for v in stats.values())
        t_acc = t_true / t_eval if t_eval else None
        return t_eval, t_true, t_false, t_acc

    tot_unique_eval, tot_unique_true, tot_unique_false, tot_unique_acc = subset_totals("unique")
    tot_multi_eval, tot_multi_true, tot_multi_false, tot_multi_acc = subset_totals("multiple")

    csv_path = osp.join(out_root, f"metric_llm_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sceneID",
                "total_queries_evaluated",
                "true",
                "false",
                "accuracy",
                "skipped",
                "target_missing",
                "target_missing_rate",
                "unique_evaluated",
                "unique_true",
                "unique_false",
                "unique_accuracy",
                "multiple_evaluated",
                "multiple_true",
                "multiple_false",
                "multiple_accuracy",
            ]
        )
        writer.writerow(
            [
                "total",
                total_eval,
                total_true,
                total_false,
                f"{total_acc:.4f}" if total_acc is not None else "",
                total_skipped,
                total_target_missing,
                f"{total_target_missing_rate:.4f}" if total_target_missing_rate is not None else "",
                tot_unique_eval,
                tot_unique_true,
                tot_unique_false,
                f"{tot_unique_acc:.4f}" if tot_unique_acc is not None else "",
                tot_multi_eval,
                tot_multi_true,
                tot_multi_false,
                f"{tot_multi_acc:.4f}" if tot_multi_acc is not None else "",
            ]
        )
        for scene_id in sorted(stats):
            s = stats[scene_id]
            ev = s["evaluated"]
            tr = s["true"]
            fl = s["false"]
            acc = tr / ev if ev else None
            tmiss = s["target_missing"]
            tmiss_rate = tmiss / ev if ev else None
            u = s["subsets"]["unique"]
            m = s["subsets"]["multiple"]
            u_acc = u["true"] / u["evaluated"] if u["evaluated"] else None
            m_acc = m["true"] / m["evaluated"] if m["evaluated"] else None
            writer.writerow(
                [
                    scene_id,
                    ev,
                    tr,
                    fl,
                    f"{acc:.4f}" if acc is not None else "",
                    s["skipped"],
                    tmiss,
                    f"{tmiss_rate:.4f}" if tmiss_rate is not None else "",
                    u["evaluated"],
                    u["true"],
                    u["false"],
                    f"{u_acc:.4f}" if u_acc is not None else "",
                    m["evaluated"],
                    m["true"],
                    m["false"],
                    f"{m_acc:.4f}" if m_acc is not None else "",
                ]
            )

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    total_acc_str = f"{total_acc:.4f}" if total_acc is not None else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {csv_path}. Accuracy={total_acc_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )

    if getattr(args, "compute_metrics", True):
        _shutdown_llm()
        eval_script = osp.join(osp.dirname(__file__), "eval", "eval_scanrefer_iou.py")
        cmd = [sys.executable, eval_script, "--run-dir", out_root, "--split", split]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[warn] IoU evaluation failed: {e}")


def run_scanrefer(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_scanrefer_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_scanrefer_split(cfg, args, split=args.split)


def _run_reason3d_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_dataset(cfg, split)
    if any("type" in q for q in queries):
        queries = [q for q in queries if q.get("type") == "scannet"]
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {"timestamp": run_ts, "script_args": script_args, "graph_root_resolved": graph_root, "split": split}
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    per_query: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, float]] = {}
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    skipped_missing_graph = 0

    def ensure_scene(scene_id: str) -> Dict[str, float]:
        if scene_id not in stats:
            stats[scene_id] = {"true": 0, "false": 0, "evaluated": 0, "skipped": 0}
        return stats[scene_id]

    def process_batch(entries: List[Dict[str, Any]]):
        nonlocal preds_written
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            scene_id = entry["scene_id"]
            graph = entry["graph"]
            s = ensure_scene(scene_id)
            pred_obj_id = result.get("prediction")
            parse_fail = result.get("parse_fail", False)
            pred_id_int = None
            if not parse_fail and pred_obj_id is not None:
                try:
                    pred_id_int = int(pred_obj_id)
                except Exception:
                    pred_id_int = None
            pred_class_id = None
            if pred_id_int is not None:
                meta = graph.get("object_metadata", {}).get(str(pred_id_int)) or graph.get("object_metadata", {}).get(
                    pred_id_int, {}
                )
                pred_class_id = meta.get("class_id") if meta else None
            target_class_id = entry["target_class_id"]
            correct = (
                pred_class_id is not None
                and target_class_id is not None
                and int(pred_class_id) == int(target_class_id)
            )
            s["evaluated"] += 1
            if correct:
                s["true"] += 1
            else:
                s["false"] += 1

            out_obj = {
                "scene": scene_id,
                "query": entry["query_text"],
                "prediction": pred_id_int,
                "prediction_class_id": pred_class_id,
                "target_class_id": target_class_id,
                "correct": correct,
                "parse_fail": parse_fail,
            }
            if args.allow_rationale:
                out_obj["rationale"] = result.get("rationale", "")
            if args.save_per_query or args.scene:
                per_query.append(out_obj)
                prefix = f"{output_prefix}_" if output_prefix else ""
                out_path = osp.join(out_root, f"{prefix}{scene_id}_q{entry['idx']}.json")
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
            pred_row = {
                "scene_id": scene_id,
                "idx": entry["idx"],
                "gt_object_id": target_class_id,
                "prediction": pred_id_int,
                "parse_fail": parse_fail,
                "skipped": False,
                "data_type": entry.get("data_type"),
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
        entries.clear()

    scene_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, q in enumerate(queries):
        scene_groups.setdefault(q["scene_id"], []).append((idx, q))

    for scene_id in tqdm.tqdm(sorted(scene_groups.keys()), desc="scenes"):
        scene_queries = scene_groups[scene_id]
        s = ensure_scene(scene_id)
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += len(scene_queries)
            s["skipped"] += len(scene_queries)
            for orig_idx, q in scene_queries:
                tgt = q.get("object_id")
                try:
                    tgt_int = int(tgt)
                except Exception:
                    tgt_int = None
                pred_row = {
                    "scene_id": scene_id,
                    "idx": orig_idx,
                    "gt_object_id": tgt_int,
                    "prediction": None,
                    "parse_fail": False,
                    "skipped": True,
                    "data_type": q.get("type"),
                }
                preds_f.write(json.dumps(pred_row) + "\n")
                preds_written += 1
                if preds_written % preds_flush_every == 0:
                    preds_f.flush()
            continue
        graph = load_graph(graph_root, scene_id)
        enrich_class_ids(graph)

        target_ids_all: List[int] = []
        for _, q in scene_queries:
            tgt = q.get("object_id")  # class id target
            try:
                tgt_int = int(tgt)
            except Exception:
                tgt_int = None
            target_ids_all.append(None)  # no object-id target for trimming

        keep_ids = trim_objects(graph, args.max_objects, target_ids_all)

        entries: List[Dict[str, Any]] = []
        for orig_idx, q in scene_queries:
            query_text = q.get("query") or q.get("question") or q.get("description") or ""
            prompt = format_prompt_selection(
                scene_id,
                graph,
                query_text,
                cfg["prompt"]["instruction"],
                keep_ids,
                include_edges=args.use_edges,
                neighbor_cap=args.neighbor_cap,
                prompt_format=args.prompt_format,
                use_attributes=args.use_attributes,
                use_colors=args.use_colors,
                use_size=args.use_size,
            )
            tgt_class = q.get("object_id")
            try:
                tgt_class = int(tgt_class) if tgt_class is not None else None
            except Exception:
                tgt_class = None
            entries.append(
                {
                    "idx": orig_idx,
                    "scene_id": scene_id,
                    "prompt": prompt,
                    "query_text": query_text,
                    "keep_ids": keep_ids,
                    "target_class_id": tgt_class,
                    "graph": graph,
                    "data_type": q.get("type"),
                }
            )
            if len(entries) >= args.batch_size:
                process_batch(entries)
        process_batch(entries)

    preds_f.flush()
    preds_f.close()

    total_true = sum(v["true"] for v in stats.values())
    total_false = sum(v["false"] for v in stats.values())
    total_eval = sum(v["evaluated"] for v in stats.values())
    total_acc = total_true / total_eval if total_eval else None

    csv_path = osp.join(out_root, f"metric_llm_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sceneID", "total_queries_evaluated", "true", "false", "accuracy", "skipped"])
        writer.writerow(
            [
                "total",
                total_eval,
                total_true,
                total_false,
                f"{total_acc:.4f}" if total_acc is not None else "",
                sum(v["skipped"] for v in stats.values()),
            ]
        )
        for scene_id in sorted(stats):
            s = stats[scene_id]
            ev = s["evaluated"]
            acc = s["true"] / ev if ev else None
            writer.writerow(
                [scene_id, ev, s["true"], s["false"], f"{acc:.4f}" if acc is not None else "", s["skipped"]]
            )

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    total_acc_str = f"{total_acc:.4f}" if total_acc is not None else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {csv_path}. Accuracy={total_acc_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )

    if getattr(args, "compute_metrics", True):
        _shutdown_llm()
        eval_script = osp.join(osp.dirname(__file__), "eval", "eval_reason3d_iou.py")
        cmd = [sys.executable, eval_script, "--run-dir", out_root, "--split", split]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[warn] IoU evaluation failed: {e}")


def run_reason3d(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_reason3d_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_reason3d_split(cfg, args, split=args.split)


def _run_scan2cap_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_dataset(cfg, split)
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {"timestamp": run_ts, "script_args": script_args, "graph_root_resolved": graph_root, "split": split}
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    # reference captions grouped by (scene, obj_id)
    ref_map: Dict[Tuple[str, int], List[str]] = {}
    for q in queries:
        scene_id = q.get("scene_id")
        obj_raw = q.get("object_id")
        try:
            obj_int = int(obj_raw) + cfg["data"]["id_offset"]
        except Exception:
            continue
        ref_map.setdefault((scene_id, obj_int), []).append(q.get("description", ""))

    per_query: List[Dict[str, Any]] = []
    caption_preds: List[str] = []
    caption_refs: List[List[str]] = []
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    skipped_missing_graph = 0
    def process_batch(entries: List[Dict[str, Any]]):
        nonlocal preds_written
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            cap_pred = _coerce_prediction_text(result.get("prediction"))
            cap_pred = str(cap_pred)[:256]
            caption_preds.append(cap_pred)
            caption_refs.append(entry["refs_gt"])
            parse_fail = bool(result.get("parse_fail", False))
            out_obj = {
                "scene": entry["scene_id"],
                "object_id_0idx": entry["target_obj_raw"],
                "object_id_1idx": entry["target_obj_id"],
                "caption_pred": cap_pred,
                "refs_gt": entry["refs_gt"],
                "query_idx": entry["idx"],
            }
            if args.allow_rationale:
                out_obj["rationale"] = result.get("rationale", "")
            pred_row = {
                "scene_id": entry["scene_id"],
                "idx": entry["idx"],
                "object_id_0idx": entry["target_obj_raw"],
                "object_id_1idx": entry["target_obj_id"],
                "prediction": cap_pred,
                "gt_captions": entry["refs_gt"],
                "parse_fail": parse_fail,
                "skipped": False,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            if args.save_per_query or args.scene:
                per_query.append(out_obj)
                prefix = f"{output_prefix}_" if output_prefix else ""
                out_path = osp.join(out_root, f"{prefix}{entry['scene_id']}_q{entry['idx']}.json")
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
        entries.clear()

    batch_entries: List[Dict[str, Any]] = []
    for idx, q in enumerate(tqdm.tqdm(queries, desc="queries")):
        scene_id = q["scene_id"]
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += 1
            target_obj_id = q.get("object_id")
            try:
                target_obj_id_int = int(target_obj_id) + cfg["data"]["id_offset"]
            except Exception:
                target_obj_id_int = None
            refs_gt = ref_map.get((scene_id, target_obj_id_int), [q.get("description", "")])
            pred_row = {
                "scene_id": scene_id,
                "idx": idx,
                "object_id_0idx": target_obj_id,
                "object_id_1idx": target_obj_id_int,
                "prediction": None,
                "gt_captions": refs_gt,
                "parse_fail": False,
                "skipped": True,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            continue
        graph = load_graph(graph_root, scene_id)

        target_obj_id = q.get("object_id")
        try:
            target_obj_id_int = int(target_obj_id) + cfg["data"]["id_offset"]
        except Exception:
            target_obj_id_int = None

        keep_ids = trim_objects(graph, args.max_objects, [target_obj_id_int])
        prompt = format_prompt_caption(
            scene_id,
            graph,
            target_obj_id_int,
            cfg["prompt"]["instruction"],
            keep_ids,
            include_edges=args.use_edges,
            neighbor_cap=args.neighbor_cap,
            prompt_format=args.prompt_format,
            use_attributes=args.use_attributes,
            use_colors=args.use_colors,
            use_size=args.use_size,
        )
        refs_gt = ref_map.get((scene_id, target_obj_id_int), [q.get("description", "")])
        batch_entries.append(
            {
                "idx": idx,
                "scene_id": scene_id,
                "prompt": prompt,
                "keep_ids": keep_ids,
                "target_obj_raw": target_obj_id,
                "target_obj_id": target_obj_id_int,
                "refs_gt": refs_gt,
            }
        )
        if len(batch_entries) >= args.batch_size:
            process_batch(batch_entries)
    process_batch(batch_entries)

    preds_f.flush()
    preds_f.close()

    cap_metrics = None
    if getattr(args, "compute_metrics", True):
        try:
            cap_metrics = _compute_text_metrics(caption_preds, caption_refs)
        except Exception as e:
            print(f"[warn] Caption metric computation failed: {e}")
            cap_metrics = None

    metrics_path = osp.join(out_root, f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump({"caption_metrics": cap_metrics, "skipped_missing_graph": skipped_missing_graph}, f, indent=2)

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    cider_str = f"{cap_metrics.get('Cider', 0.0):.4f}" if cap_metrics else "n/a"
    bleu_str = f"{cap_metrics.get('Bleu', 0.0):.4f}" if cap_metrics else "n/a"
    meteor_str = f"{cap_metrics.get('Meteor', 0.0):.4f}" if cap_metrics else "n/a"
    rouge_str = f"{cap_metrics.get('RougeL', 0.0):.4f}" if cap_metrics else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {metrics_path}. "
        f"CIDEr={cider_str}, BLEU-4={bleu_str}, METEOR={meteor_str}, ROUGE-L={rouge_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )


def run_scan2cap(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_scan2cap_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_scan2cap_split(cfg, args, split=args.split)


def _run_scanqa_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_dataset(cfg, split)
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {"timestamp": run_ts, "script_args": script_args, "graph_root_resolved": graph_root, "split": split}
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    per_query: List[Dict[str, Any]] = []
    answer_preds: List[str] = []
    answer_refs: List[List[str]] = []
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    skipped_missing_graph = 0
    def process_batch(entries: List[Dict[str, Any]]):
        nonlocal preds_written
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            pred_text = _coerce_prediction_text(result.get("prediction"))
            ans = pred_text.strip() if isinstance(pred_text, str) else str(pred_text)
            rat = result.get("rationale", "") if args.allow_rationale else ""
            answer_preds.append(ans)
            answer_refs.append(entry["refs_gt"])
            parse_fail = bool(result.get("parse_fail", False))
            out_obj = {
                "scene": entry["scene_id"],
                "question": entry["question"],
                "answer_pred": ans,
                "refs_gt": entry["refs_gt"],
                "query_idx": entry["idx"],
            }
            if args.allow_rationale:
                out_obj["rationale"] = rat
            pred_row = {
                "scene_id": entry["scene_id"],
                "idx": entry["idx"],
                "question": entry["question"],
                "prediction": ans,
                "gt_answers": entry["refs_gt"],
                "parse_fail": parse_fail,
                "skipped": False,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            if args.save_per_query or args.scene:
                per_query.append(out_obj)
                prefix = f"{output_prefix}_" if output_prefix else ""
                out_path = osp.join(out_root, f"{prefix}{entry['scene_id']}_q{entry['idx']}.json")
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
        entries.clear()

    batch_entries: List[Dict[str, Any]] = []
    for idx, q in enumerate(tqdm.tqdm(queries, desc="queries")):
        scene_id = q["scene_id"]
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += 1
            question = q.get("question") or q.get("query") or ""
            refs_gt = q.get("answers") or q.get("answer_choices") or []
            if isinstance(refs_gt, str):
                refs_gt = [refs_gt]
            if not refs_gt:
                refs_gt = [""]
            pred_row = {
                "scene_id": scene_id,
                "idx": idx,
                "question": question,
                "prediction": None,
                "gt_answers": refs_gt,
                "parse_fail": False,
                "skipped": True,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            continue
        graph = load_graph(graph_root, scene_id)

        # Some ScanQA samples have target object ids; keep for trimming if present.
        obj_ids_raw = q.get("object_ids") or q.get("object_id") or []
        if not isinstance(obj_ids_raw, list):
            obj_ids_raw = [obj_ids_raw]
        try:
            target_ids = [int(x) + cfg["data"]["id_offset"] for x in obj_ids_raw if x is not None]
        except Exception:
            target_ids = []
        keep_ids = trim_objects(graph, args.max_objects, target_ids)

        question = q.get("question") or q.get("query") or ""
        prompt = format_prompt_qa(
            scene_id,
            graph,
            question,
            cfg["prompt"]["instruction"],
            keep_ids,
            include_edges=args.use_edges,
            neighbor_cap=args.neighbor_cap,
            prompt_format=args.prompt_format,
            use_attributes=args.use_attributes,
            use_colors=args.use_colors,
            use_size=args.use_size,
        )
        refs_gt = q.get("answers") or q.get("answer_choices") or []
        if isinstance(refs_gt, str):
            refs_gt = [refs_gt]
        if not refs_gt:
            refs_gt = [""]

        batch_entries.append(
            {
                "idx": idx,
                "scene_id": scene_id,
                "prompt": prompt,
                "keep_ids": keep_ids,
                "question": question,
                "refs_gt": refs_gt,
            }
        )
        if len(batch_entries) >= args.batch_size:
            process_batch(batch_entries)
    process_batch(batch_entries)

    preds_f.flush()
    preds_f.close()

    qa_metrics = None
    if getattr(args, "compute_metrics", True):
        try:
            qa_metrics = _compute_text_metrics(answer_preds, answer_refs)
        except Exception as e:
            print(f"[warn] QA metric computation failed: {e}")
            qa_metrics = None

    metrics_path = osp.join(out_root, f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump({"qa_metrics": qa_metrics, "skipped_missing_graph": skipped_missing_graph}, f, indent=2)

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    cider_str = f"{qa_metrics.get('Cider', 0.0):.4f}" if qa_metrics else "n/a"
    bleu_str = f"{qa_metrics.get('Bleu', 0.0):.4f}" if qa_metrics else "n/a"
    meteor_str = f"{qa_metrics.get('Meteor', 0.0):.4f}" if qa_metrics else "n/a"
    rouge_str = f"{qa_metrics.get('RougeL', 0.0):.4f}" if qa_metrics else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {metrics_path}. "
        f"CIDEr={cider_str}, BLEU-4={bleu_str}, METEOR={meteor_str}, ROUGE-L={rouge_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )


def run_scanqa(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_scanqa_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_scanqa_split(cfg, args, split=args.split)


def _run_sqa3d_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_sqa3d_dataset(cfg, split)
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {"timestamp": run_ts, "script_args": script_args, "graph_root_resolved": graph_root, "split": split}
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    per_query: List[Dict[str, Any]] = []
    answer_preds: List[str] = []
    answer_refs: List[List[str]] = []
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    skipped_missing_graph = 0

    def process_batch(entries: List[Dict[str, Any]]):
        nonlocal preds_written
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            pred_text = _coerce_prediction_text(result.get("prediction"))
            ans = pred_text.strip() if isinstance(pred_text, str) else str(pred_text)
            rat = result.get("rationale", "") if args.allow_rationale else ""
            answer_preds.append(ans)
            answer_refs.append(entry["refs_gt"])
            parse_fail = bool(result.get("parse_fail", False))
            out_obj = {
                "scene": entry["scene_id"],
                "question": entry["question_raw"],
                "answer_pred": ans,
                "refs_gt": entry["refs_gt"],
                "query_idx": entry["idx"],
                "question_id": entry.get("question_id"),
            }
            if args.allow_rationale:
                out_obj["rationale"] = rat
            pred_row = {
                "scene_id": entry["scene_id"],
                "idx": entry["idx"],
                "question_id": entry.get("question_id"),
                "question": entry["question_raw"],
                "prediction": ans,
                "gt_answers": entry["refs_gt"],
                "parse_fail": parse_fail,
                "skipped": False,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            if args.save_per_query or args.scene:
                per_query.append(out_obj)
                prefix = f"{output_prefix}_" if output_prefix else ""
                out_path = osp.join(out_root, f"{prefix}{entry['scene_id']}_q{entry['idx']}.json")
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
        entries.clear()

    batch_entries: List[Dict[str, Any]] = []
    for idx, q in enumerate(tqdm.tqdm(queries, desc="queries")):
        scene_id = q.get("scene_id")
        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += 1
            question = q.get("question", "")
            situation = q.get("situation", "")
            question_raw = f"Situation: {situation} Question: {question}".strip()
            refs_gt = q.get("answers") or []
            if isinstance(refs_gt, str):
                refs_gt = [refs_gt]
            if not refs_gt:
                refs_gt = [""]
            pred_row = {
                "scene_id": scene_id,
                "idx": idx,
                "question_id": q.get("question_id"),
                "question": question_raw,
                "prediction": None,
                "gt_answers": refs_gt,
                "parse_fail": False,
                "skipped": True,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            continue
        graph = load_graph(graph_root, scene_id)

        keep_ids = trim_objects(graph, args.max_objects, [])
        situation = q.get("situation", "")
        question = q.get("question", "")
        if situation:
            question_raw = f"Situation: {situation} Question: {question}"
        else:
            question_raw = question
        prompt = format_prompt_qa(
            scene_id,
            graph,
            question_raw,
            cfg["prompt"]["instruction"],
            keep_ids,
            include_edges=args.use_edges,
            neighbor_cap=args.neighbor_cap,
            prompt_format=args.prompt_format,
            use_attributes=args.use_attributes,
            use_colors=args.use_colors,
            use_size=args.use_size,
        )
        refs_gt = q.get("answers") or []
        if isinstance(refs_gt, str):
            refs_gt = [refs_gt]
        if not refs_gt:
            refs_gt = [""]

        batch_entries.append(
            {
                "idx": idx,
                "scene_id": scene_id,
                "question_id": q.get("question_id"),
                "prompt": prompt,
                "question_raw": question_raw,
                "refs_gt": refs_gt,
            }
        )
        if len(batch_entries) >= args.batch_size:
            process_batch(batch_entries)
    process_batch(batch_entries)

    preds_f.flush()
    preds_f.close()

    qa_metrics = None
    em1 = None
    if getattr(args, "compute_metrics", True):
        try:
            qa_metrics = _compute_text_metrics(answer_preds, answer_refs)
        except Exception as e:
            print(f"[warn] QA metric computation failed: {e}")
            qa_metrics = None
        try:
            em1 = _compute_sqa3d_em1(answer_preds, answer_refs)
        except Exception as e:
            print(f"[warn] EM1 computation failed: {e}")
            em1 = None

    metrics_path = osp.join(out_root, f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump({"qa_metrics": qa_metrics, "em1": em1, "skipped_missing_graph": skipped_missing_graph}, f, indent=2)

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    cider_str = f"{qa_metrics.get('Cider', 0.0):.4f}" if qa_metrics else "n/a"
    bleu_str = f"{qa_metrics.get('Bleu', 0.0):.4f}" if qa_metrics else "n/a"
    meteor_str = f"{qa_metrics.get('Meteor', 0.0):.4f}" if qa_metrics else "n/a"
    rouge_str = f"{qa_metrics.get('RougeL', 0.0):.4f}" if qa_metrics else "n/a"
    em1_str = f"{em1:.4f}" if em1 is not None else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {metrics_path}. "
        f"EM1={em1_str}, CIDEr={cider_str}, BLEU-4={bleu_str}, METEOR={meteor_str}, ROUGE-L={rouge_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )


def run_sqa3d(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_sqa3d_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_sqa3d_split(cfg, args, split=args.split)


def _run_surprise3d_split(
    cfg: Dict[str, Any],
    args: SimpleNamespace,
    split: str,
    run_ts: Optional[str] = None,
    out_root: Optional[str] = None,
    output_prefix: Optional[str] = None,
):
    queries = load_dataset(cfg, split)
    if args.scene:
        queries = [q for q in queries if q.get("scene_id") == args.scene]
    if args.max_queries:
        queries = queries[: args.max_queries]

    graph_root = args.graph_root
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = out_root or osp.join(args.out_root, args.graph, run_ts)
    os.makedirs(out_root, exist_ok=True)

    script_args = _build_script_args(args)
    script_args["split"] = split
    params_name = f"all_run_params_{output_prefix}.json" if output_prefix else "all_run_params.json"
    all_run_params = {
        "timestamp": run_ts,
        "script_args": script_args,
        "graph_root_resolved": graph_root,
        "split": split,
    }
    with open(osp.join(out_root, params_name), "w") as f:
        json.dump(all_run_params, f, indent=2)

    preds_path = osp.join(out_root, f"preds_{split}.jsonl")
    preds_f = open(preds_path, "w", buffering=1)
    preds_written = 0
    preds_flush_every = 200

    stats_scene: Dict[str, Dict[str, Any]] = {}
    stats_qtype: Dict[str, Dict[str, Any]] = {}
    stats_total = {
        "evaluated": 0,
        "skipped": 0,
        "coverage_sum": 0.0,
        "jaccard_sum": 0.0,
        "any_hit_true": 0,
        "exact_true": 0,
        "gt_count_sum": 0,
        "pred_count_sum": 0,
    }
    token_totals: Dict[str, int] = {"input": 0, "output": 0, "samples": 0}
    stats_totals: Dict[str, float] = _init_stats_totals()
    timing_totals: Dict[str, float] = {"time_sec": 0.0, "samples": 0}
    per_query: List[Dict[str, Any]] = []
    skipped_missing_graph = 0
    qtype_labels = {
        "cs": "Common-sense",
        "hi": "Human Intention",
        "first_view": "Narrative Perspective",
        "camera_view": "Parametric Perspective",
        "relative_position": "Relative Position",
        "abs": "Absolute Distance",
    }
    qtype_order = ["cs", "hi", "first_view", "camera_view", "relative_position", "abs"]

    def _init_stats():
        return {
            "evaluated": 0,
            "skipped": 0,
            "coverage_sum": 0.0,
            "jaccard_sum": 0.0,
            "any_hit_true": 0,
            "exact_true": 0,
            "gt_count_sum": 0,
            "pred_count_sum": 0,
        }

    def ensure_scene(scene_id: str) -> Dict[str, Any]:
        if scene_id not in stats_scene:
            stats_scene[scene_id] = _init_stats()
        return stats_scene[scene_id]

    def ensure_qtype(qtype: str) -> Dict[str, Any]:
        if qtype not in stats_qtype:
            stats_qtype[qtype] = _init_stats()
        return stats_qtype[qtype]

    def evaluate_sets(gt_ids: List[int], pred_ids: List[int]) -> Tuple[float, float, int, int]:
        gt_set = set(gt_ids)
        pred_set = set(pred_ids)
        if not gt_set and not pred_set:
            return 1.0, 1.0, 1, 1
        intersection = len(gt_set & pred_set)
        union = len(gt_set | pred_set)
        coverage = intersection / len(gt_set) if gt_set else 0.0
        jaccard = intersection / union if union else 0.0
        any_hit = 1 if intersection > 0 else 0
        exact = 1 if gt_set == pred_set and gt_set else 0
        return coverage, jaccard, any_hit, exact

    def process_batch(entries: List[Dict[str, Any]]):
        nonlocal preds_written
        if not entries:
            return
        t0 = time.perf_counter()
        batch_results = call_model_batch_vllm(entries, args, token_totals, stats_totals)
        timing_totals["time_sec"] = timing_totals.get("time_sec", 0.0) + (time.perf_counter() - t0)
        timing_totals["samples"] = timing_totals.get("samples", 0) + len(entries)
        for entry, result in zip(entries, batch_results):
            scene_id = entry["scene_id"]
            s = ensure_scene(scene_id)
            qt = entry.get("question_type") or "unknown"
            qt_stat = ensure_qtype(qt)
            pred_ids = _normalize_id_list(result.get("prediction"))
            parse_fail = bool(result.get("parse_fail", False))
            gt_ids = entry["gt_ids_eval"]
            coverage, jaccard, any_hit, exact = evaluate_sets(gt_ids, pred_ids)
            s["evaluated"] += 1
            stats_total["evaluated"] += 1
            s["coverage_sum"] += coverage
            s["jaccard_sum"] += jaccard
            s["any_hit_true"] += any_hit
            s["exact_true"] += exact
            s["gt_count_sum"] += len(gt_ids)
            s["pred_count_sum"] += len(pred_ids)
            stats_total["coverage_sum"] += coverage
            stats_total["jaccard_sum"] += jaccard
            stats_total["any_hit_true"] += any_hit
            stats_total["exact_true"] += exact
            stats_total["gt_count_sum"] += len(gt_ids)
            stats_total["pred_count_sum"] += len(pred_ids)
            qt_stat["evaluated"] += 1
            qt_stat["coverage_sum"] += coverage
            qt_stat["jaccard_sum"] += jaccard
            qt_stat["any_hit_true"] += any_hit
            qt_stat["exact_true"] += exact
            qt_stat["gt_count_sum"] += len(gt_ids)
            qt_stat["pred_count_sum"] += len(pred_ids)
            out_obj = {
                "scene": scene_id,
                "question_type": entry["question_type"],
                "question": entry["query_text"],
                "prediction": pred_ids,
                "gt_object_ids_1idx": gt_ids,
                "gt_object_ids_0idx": entry["gt_ids_raw"],
                "coverage": coverage,
                "jaccard": jaccard,
                "any_hit": any_hit,
                "exact": exact,
                "query_idx": entry["idx"],
            }
            if args.allow_rationale:
                out_obj["rationale"] = result.get("rationale", "")
            if args.save_per_query or args.scene:
                per_query.append(out_obj)
                out_path = osp.join(out_root, f"{scene_id}_q{entry['idx']}.json")
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
            pred_row = {
                "scene_id": scene_id,
                "idx": entry["idx"],
                "gt_object_ids": entry["gt_ids_raw"],
                "gt_object_ids_eval": gt_ids,
                "prediction": pred_ids,
                "parse_fail": parse_fail,
                "skipped": False,
                "question_type": entry["question_type"],
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
        entries.clear()

    batch_entries: List[Dict[str, Any]] = []
    for idx, q in enumerate(tqdm.tqdm(queries, desc="queries")):
        scene_id = q["scene_id"]
        question_type = q.get("question_type") or q.get("eval_type") or "unknown"

        gt_ids_raw = q.get("object_id") or q.get("object_ids") or []
        if not isinstance(gt_ids_raw, list):
            gt_ids_raw = [gt_ids_raw]
        gt_ids_raw_int = []
        for v in gt_ids_raw:
            try:
                gt_ids_raw_int.append(int(v))
            except Exception:
                continue
        gt_ids_eval = [i + cfg["data"]["id_offset"] for i in gt_ids_raw_int]

        graph_path = osp.join(graph_root, f"{scene_id}.json")
        if not osp.exists(graph_path):
            skipped_missing_graph += 1
            s = ensure_scene(scene_id)
            qt_stat = ensure_qtype(question_type)
            s["skipped"] += 1
            qt_stat["skipped"] += 1
            stats_total["skipped"] += 1
            pred_row = {
                "scene_id": scene_id,
                "idx": idx,
                "gt_object_ids": gt_ids_raw_int,
                "gt_object_ids_eval": gt_ids_eval,
                "prediction": None,
                "parse_fail": False,
                "skipped": True,
                "question_type": question_type,
            }
            preds_f.write(json.dumps(pred_row) + "\n")
            preds_written += 1
            if preds_written % preds_flush_every == 0:
                preds_f.flush()
            continue
        graph = load_graph(graph_root, scene_id)
        subset_map = build_subset_map(graph)

        keep_ids = trim_objects(graph, args.max_objects, gt_ids_eval)
        subset = subset_map.get(gt_ids_eval[0]) if gt_ids_eval else "unknown"

        query_text = q.get("description") or q.get("question") or q.get("query") or ""
        prompt = format_prompt_selection(
            scene_id,
            graph,
            query_text,
            cfg["prompt"]["instruction"],
            keep_ids,
            include_edges=args.use_edges,
            neighbor_cap=args.neighbor_cap,
            question_type=question_type,
            prompt_format=args.prompt_format,
            use_attributes=args.use_attributes,
            use_colors=args.use_colors,
            use_size=args.use_size,
        )
        batch_entries.append(
            {
                "idx": idx,
                "scene_id": scene_id,
                "prompt": prompt,
                "keep_ids": keep_ids,
                "gt_ids_eval": gt_ids_eval,
                "gt_ids_raw": gt_ids_raw_int,
                "query_text": query_text,
                "subset": subset,
                "question_type": question_type,
            }
        )
        if len(batch_entries) >= args.batch_size:
            process_batch(batch_entries)
    process_batch(batch_entries)
    preds_f.flush()
    preds_f.close()

    def finalize_stats(stat: Dict[str, Any]) -> Dict[str, Any]:
        evaluated = stat["evaluated"]
        def safe_div(num):
            return num / evaluated if evaluated else None
        return {
            "evaluated": evaluated,
            "skipped": stat["skipped"],
            "coverage": safe_div(stat["coverage_sum"]),
            "jaccard": safe_div(stat["jaccard_sum"]),
            "any_hit": safe_div(stat["any_hit_true"]),
            "exact": safe_div(stat["exact_true"]),
            "avg_gt_count": stat["gt_count_sum"] / evaluated if evaluated else None,
            "avg_pred_count": stat["pred_count_sum"] / evaluated if evaluated else None,
        }

    csv_path = osp.join(out_root, f"metric_llm_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_type",
                "evaluated",
                "skipped",
                "coverage",
                "jaccard",
                "any_hit",
                "exact",
                "avg_gt_count",
                "avg_pred_count",
            ]
        )
        ordered_qtypes = [q for q in qtype_order if q in stats_qtype]
        other_qtypes = sorted(q for q in stats_qtype if q not in qtype_order)
        for qtype in ordered_qtypes + other_qtypes:
            s = finalize_stats(stats_qtype[qtype])
            display_qt = qtype_labels.get(qtype, qtype)
            writer.writerow(
                [
                    display_qt,
                    s["evaluated"],
                    s["skipped"],
                    f"{s['coverage']:.4f}" if s["coverage"] is not None else "",
                    f"{s['jaccard']:.4f}" if s["jaccard"] is not None else "",
                    f"{s['any_hit']:.4f}" if s["any_hit"] is not None else "",
                    f"{s['exact']:.4f}" if s["exact"] is not None else "",
                    f"{s['avg_gt_count']:.2f}" if s["avg_gt_count"] is not None else "",
                    f"{s['avg_pred_count']:.2f}" if s["avg_pred_count"] is not None else "",
                ]
            )
        total_fmt = finalize_stats(stats_total)
        writer.writerow(
            [
                "overall",
                total_fmt["evaluated"],
                total_fmt["skipped"],
                f"{total_fmt['coverage']:.4f}" if total_fmt["coverage"] is not None else "",
                f"{total_fmt['jaccard']:.4f}" if total_fmt["jaccard"] is not None else "",
                f"{total_fmt['any_hit']:.4f}" if total_fmt["any_hit"] is not None else "",
                f"{total_fmt['exact']:.4f}" if total_fmt["exact"] is not None else "",
                f"{total_fmt['avg_gt_count']:.2f}" if total_fmt["avg_gt_count"] is not None else "",
                f"{total_fmt['avg_pred_count']:.2f}" if total_fmt["avg_pred_count"] is not None else "",
            ]
        )

    total_samples = token_totals.get("samples", 0)
    avg_input_tokens = (token_totals["input"] / total_samples) if total_samples else None
    avg_output_tokens = (token_totals["output"] / total_samples) if total_samples else None
    avg_latency = (timing_totals["time_sec"] / timing_totals["samples"]) if timing_totals["samples"] else None
    engine_metrics = _collect_vllm_engine_metrics()
    _write_run_metrics_csv(out_root, split, token_totals, timing_totals, stats_totals, engine_metrics)
    coverage_str = f"{total_fmt['coverage']:.4f}" if total_fmt["coverage"] is not None else "n/a"
    avg_in_str = f"{avg_input_tokens:.1f}" if avg_input_tokens is not None else "n/a"
    avg_out_str = f"{avg_output_tokens:.1f}" if avg_output_tokens is not None else "n/a"
    avg_lat_str = f"{avg_latency*1000:.1f}ms" if avg_latency is not None else "n/a"
    print(
        f"Done. Saved summary to {csv_path}. Coverage={coverage_str}. "
        f"Avg input tokens={avg_in_str}, avg output tokens={avg_out_str}, avg latency per query={avg_lat_str}"
    )

    if getattr(args, "compute_metrics", True):
        _shutdown_llm()
        eval_cfg = cfg.get("evaluation", {}) or {}
        run_iou = eval_cfg.get("run_surprise3d_iou", True)
        if run_iou:
            eval_script = osp.join(osp.dirname(__file__), "eval", "eval_surprise3d_iou.py")
            cmd = [sys.executable, eval_script, "--run-dir", out_root, "--split", split]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"[warn] IoU evaluation failed: {e}")

        if eval_cfg.get("run_f1", False):
            f1_script = eval_cfg.get("f1_script")
            if not f1_script:
                f1_script = osp.join(osp.dirname(__file__), "eval", "eval_multi3drefer_f1.py")
            if not osp.isabs(f1_script) and not osp.exists(f1_script):
                repo_root = osp.dirname(osp.dirname(osp.dirname(__file__)))
                candidate = osp.join(repo_root, f1_script)
                if osp.exists(candidate):
                    f1_script = candidate
            cmd = [sys.executable, f1_script, "--run-dir", out_root, "--split", split]
            extra = eval_cfg.get("f1_args") or []
            if isinstance(extra, list):
                cmd += [str(x) for x in extra if x is not None and str(x) != ""]
            elif isinstance(extra, str) and extra.strip():
                cmd += extra.strip().split()
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"[warn] F1 evaluation failed: {e}")


def run_surprise3d(cfg: Dict[str, Any], args: SimpleNamespace):
    if args.split == "all" and "train" in cfg["data"]["splits"] and "val" in cfg["data"]["splits"]:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.join(args.out_root, args.graph, run_ts)
        os.makedirs(out_root, exist_ok=True)
        print(f"[split] all -> running train + val; outputs in {out_root}")
        for split in ("train", "val"):
            _run_surprise3d_split(
                cfg,
                args,
                split=split,
                run_ts=run_ts,
                out_root=out_root,
                output_prefix=split,
            )
        return

    _run_surprise3d_split(cfg, args, split=args.split)
