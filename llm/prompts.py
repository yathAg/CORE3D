#!/usr/bin/env python3
"""Prompt formatting helpers for flat and nested graph descriptions."""

from typing import Any, Dict, List, Optional


def _build_object_lines_flat(
    scene_graph: Dict[str, Any],
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    mark_target: Optional[List[int]] = None,
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> List[str]:
    obj_meta = scene_graph.get("object_metadata", {}) or {}
    obj_names = scene_graph.get("objects", {}) or {}
    edge_feats = scene_graph.get("edge_features", {}) or {}
    neighbors = scene_graph.get("neighbors", {}) or {}
    rel_lookup = {}
    for rel in scene_graph.get("relationships", []) or []:
        if not isinstance(rel, (list, tuple)) or len(rel) < 4:
            continue
        src, dst, _, name = rel
        try:
            src_i = int(src)
        except Exception:
            src_i = src
        try:
            dst_i = int(dst)
        except Exception:
            dst_i = dst
        rel_lookup[(src_i, dst_i)] = name
    mark_target = set(mark_target or [])
    if not use_attributes or (not use_colors and not use_size):
        header = "Objects (id, label):"
    else:
        header_fields = ["id"]
        if use_colors:
            header_fields.append("color")
        header_fields.append("label")
        if use_size:
            header_fields.append("size")
        header = f"Objects ({', '.join(header_fields)}):"
    lines: List[str] = [header]
    label_lookup: Dict[Any, str] = {}
    for oid in keep_ids:
        o = obj_meta.get(str(oid)) or obj_meta.get(oid) or {}
        label = o.get("label", obj_names.get(str(oid), obj_names.get(oid)))
        label_lookup[oid] = label or "unknown"
    for oid in keep_ids:
        o = obj_meta.get(str(oid)) or obj_meta.get(oid) or {}
        label = label_lookup.get(oid, "unknown")
        if use_attributes and (use_colors or use_size):
            size = o.get("size")
            color = o.get("color_name")
            color_txt = f"{color}" if color else ""
            sz_txt = f"{size}" if size is not None else ""
            parts = [str(oid)]
            if use_colors:
                parts.append(color_txt)
            parts.append(label)
            if use_size:
                parts.append(sz_txt)
            lines.append(f"- {', '.join(parts)}")
        else:
            lines.append(f"- {oid}, {label}")

    if include_edges and neighbors:
        lines.append("Neighbors (directed, limited): (distance, relation, target_label)")
        rel_phrase_map = {
            "left": "left of",
            "right": "right of",
            "front": "in front of",
            "back": "behind",
            "above": "above",
            "below": "below",
            "near": "near",
        }
        rel_inverse = {
            "left": "right",
            "right": "left",
            "front": "back",
            "back": "front",
            "above": "below",
            "below": "above",
            "near": "near",
        }
        for oid in keep_ids:
            neigh_list = neighbors.get(str(oid)) or neighbors.get(oid) or []
            neigh_list = neigh_list[:neighbor_cap]
            if not neigh_list:
                continue
            formatted = []
            for n in neigh_list:
                try:
                    n_i = int(n)
                except Exception:
                    n_i = n
                ef = edge_feats.get(f"{oid}->{n}", {})
                parts = []
                dist = ef.get("distance")
                if dist is not None:
                    parts.append(f"{dist:.1f}m")
                rel_name = rel_lookup.get((oid, n_i))
                rel_name = rel_inverse.get(rel_name, rel_name) if rel_name else rel_name
                if rel_name:
                    phrase = rel_phrase_map.get(rel_name, rel_name)
                    parts.append(phrase)
                dst_label = label_lookup.get(n_i, "unknown")
                if dst_label:
                    parts.append(dst_label)
                formatted.append(f"({', '.join(parts)})" if parts else "()")
            lines.append(f"- {oid}: {', '.join(formatted)}")
    return lines


def _build_object_lines_nested(
    scene_graph: Dict[str, Any],
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    mark_target: Optional[List[int]] = None,
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> List[str]:
    obj_meta = scene_graph.get("object_metadata", {}) or {}
    obj_names = scene_graph.get("objects", {}) or {}
    edge_feats = scene_graph.get("edge_features", {}) or {}
    neighbors = scene_graph.get("neighbors", {}) or {}
    rel_lookup = {}
    for rel in scene_graph.get("relationships", []) or []:
        if not isinstance(rel, (list, tuple)) or len(rel) < 4:
            continue
        src, dst, _, name = rel
        try:
            src_i = int(src)
        except Exception:
            src_i = src
        try:
            dst_i = int(dst)
        except Exception:
            dst_i = dst
        rel_lookup[(src_i, dst_i)] = name
    mark_target = set(mark_target or [])
    if not use_attributes or (not use_colors and not use_size):
        header = "Objects (id, label):"
    else:
        header_fields = ["id"]
        if use_colors:
            header_fields.append("color")
        header_fields.append("label")
        if use_size:
            header_fields.append("size")
        header = f"Objects ({', '.join(header_fields)}):"
    lines: List[str] = [header]
    rel_phrase_map = {}
    rel_inverse = {}
    if include_edges and neighbors:
        lines.append("  Neighbors (distance, relation, target_label)")
        rel_phrase_map = {
            "left": "left of",
            "right": "right of",
            "front": "in front of",
            "back": "behind",
            "above": "above",
            "below": "below",
            "near": "near",
        }
        rel_inverse = {
            "left": "right",
            "right": "left",
            "front": "back",
            "back": "front",
            "above": "below",
            "below": "above",
            "near": "near",
        }
    label_lookup: Dict[Any, str] = {}
    for oid in keep_ids:
        o = obj_meta.get(str(oid)) or obj_meta.get(oid) or {}
        label = o.get("label", obj_names.get(str(oid), obj_names.get(oid)))
        label_lookup[oid] = label or "unknown"
    for oid in keep_ids:
        o = obj_meta.get(str(oid)) or obj_meta.get(oid) or {}
        label = label_lookup.get(oid, "unknown")
        if use_attributes and (use_colors or use_size):
            size = o.get("size")
            color = o.get("color_name")
            color_txt = f"{color}" if color else ""
            sz_txt = f"{size}" if size is not None else ""
            parts = [str(oid)]
            if use_colors:
                parts.append(color_txt)
            parts.append(label)
            if use_size:
                parts.append(sz_txt)
            lines.append(f"- {', '.join(parts)}")
        else:
            lines.append(f"- {oid}, {label}")

        if include_edges and neighbors:
            neigh_list = neighbors.get(str(oid)) or neighbors.get(oid) or []
            neigh_list = neigh_list[:neighbor_cap]
            if not neigh_list:
                continue
            for n in neigh_list:
                try:
                    n_i = int(n)
                except Exception:
                    n_i = n
                ef = edge_feats.get(f"{oid}->{n}", {})
                parts = []
                dist = ef.get("distance")
                if dist is not None:
                    parts.append(f"{dist:.1f}m")
                rel_name = rel_lookup.get((oid, n_i))
                rel_name = rel_inverse.get(rel_name, rel_name) if rel_name else rel_name
                if rel_name:
                    phrase = rel_phrase_map.get(rel_name, rel_name)
                    parts.append(phrase)
                dst_label = label_lookup.get(n_i, "unknown")
                if dst_label:
                    parts.append(dst_label)
                entry = ", ".join(parts) if parts else ""
                lines.append(f"  - {entry}" if entry else "  -")
    return lines


def build_object_lines(
    scene_graph: Dict[str, Any],
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    mark_target: Optional[List[int]] = None,
    prompt_format: str = "nested",
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> List[str]:
    if prompt_format == "flat":
        return _build_object_lines_flat(
            scene_graph,
            keep_ids,
            include_edges,
            neighbor_cap,
            mark_target,
            use_attributes=use_attributes,
            use_colors=use_colors,
            use_size=use_size,
        )
    if prompt_format == "nested":
        return _build_object_lines_nested(
            scene_graph,
            keep_ids,
            include_edges,
            neighbor_cap,
            mark_target,
            use_attributes=use_attributes,
            use_colors=use_colors,
            use_size=use_size,
        )
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def format_prompt_selection(
    scene_id: str,
    scene_graph: Dict[str, Any],
    query: str,
    instruction: str,
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    question_type: Optional[str] = None,
    prompt_format: str = "nested",
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> str:
    lines: List[str] = []
    lines.extend(
        build_object_lines(
            scene_graph,
            keep_ids,
            include_edges,
            neighbor_cap,
            prompt_format=prompt_format,
            use_attributes=use_attributes,
            use_colors=use_colors,
            use_size=use_size,
        )
    )
    lines.append("Query: " + query)
    lines.append(instruction)
    return "\n".join(lines)


def format_prompt_caption(
    scene_id: str,
    scene_graph: Dict[str, Any],
    target_id: Optional[int],
    instruction: str,
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    prompt_format: str = "nested",
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> str:
    lines: List[str] = []
    lines.extend(
        build_object_lines(
            scene_graph,
            keep_ids,
            include_edges,
            neighbor_cap,
            mark_target=[target_id] if target_id is not None else [],
            prompt_format=prompt_format,
            use_attributes=use_attributes,
            use_colors=use_colors,
            use_size=use_size,
        )
    )
    if prompt_format == "flat":
        lines.append(f"Target object id: {target_id if target_id is not None else 'unknown'}")
    lines.append(instruction)
    return "\n".join(lines)


def format_prompt_qa(
    scene_id: str,
    scene_graph: Dict[str, Any],
    question: str,
    instruction: str,
    keep_ids: List[int],
    include_edges: bool,
    neighbor_cap: int,
    prompt_format: str = "nested",
    use_attributes: bool = True,
    use_colors: bool = True,
    use_size: bool = True,
) -> str:
    lines: List[str] = []
    lines.extend(
        build_object_lines(
            scene_graph,
            keep_ids,
            include_edges,
            neighbor_cap,
            prompt_format=prompt_format,
            use_attributes=use_attributes,
            use_colors=use_colors,
            use_size=use_size,
        )
    )
    lines.append("Question: " + question)
    lines.append(instruction)
    return "\n".join(lines)
