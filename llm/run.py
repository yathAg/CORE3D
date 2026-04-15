#!/usr/bin/env python3
"""Unified entry point for LLM inference with flat/nested prompts and optional LoRA."""

import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace
from typing import Any, Dict, Optional

from llm import benchmarks
from llm.core import _build_structured_outputs_params


def parse_args() -> argparse.Namespace:
    def _parse_bool(value: str) -> bool:
        val = value.strip().lower()
        if val in ("1", "true", "yes", "y", "t"):
            return True
        if val in ("0", "false", "no", "n", "f"):
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")

    ap = argparse.ArgumentParser(description="Unified vLLM graph runner")
    ap.add_argument("--config", default=None, help="Path to benchmark config JSON")
    ap.add_argument("--graph", default=None, help="Graph key from config to use (overrides graph.default)")
    ap.add_argument("--graph-root", dest="graph_root", default=None, help="Override graph root directory")
    ap.add_argument("--split", default=None, help="Dataset split to run (overrides config)")
    ap.add_argument("--scene", default=None, help="Optional single-scene filter")
    ap.add_argument("--max-queries", type=int, default=None, help="Optional cap on number of queries after filtering")
    ap.add_argument("--output_path", dest="out_root", default=None, help="Override output root from config")
    ap.add_argument("--use-edges", type=_parse_bool, default=None, help="Override graph.use_edges (true/false)")
    ap.add_argument("--use-attributes", type=_parse_bool, default=None, help="Override graph.use_attributes (true/false)")
    ap.add_argument("--neighbor-cap", type=int, default=None, help="Override graph.neighbor_cap")
    ap.add_argument("--model", default=None, help="Override model id from config")
    ap.add_argument("--dtype", default=None, help="Override model dtype from config")
    ap.add_argument(
        "--quantization",
        default=None,
        help="Override quantization from config (e.g., 'fp8', 'awq', 'gptq', or 'none')",
    )
    ap.add_argument(
        "--enable-prefix-caching",
        type=_parse_bool,
        default=None,
        help="Override model.enable_prefix_caching (true/false)",
    )
    ap.add_argument("--llm-log-stats", action="store_true", help="Enable vLLM per-request progress stats (off by default)")
    ap.add_argument(
        "--prompt-format",
        default="nested",
        choices=["flat", "nested"],
        help="Prompt format to use (flat or nested). Default: nested",
    )
    ap.add_argument("--lora-path", default=None, help="Path to LoRA adapter directory (PEFT)")
    ap.add_argument("--lora-name", default=None, help="LoRA adapter name (default: folder name)")
    ap.add_argument("--lora-id", type=int, default=None, help="LoRA adapter integer id (default: 1)")
    ap.add_argument("--max-lora-rank", type=int, default=None, help="Max LoRA rank for vLLM (default: 16)")
    ap.add_argument("--max-loras", type=int, default=None, help="Max number of LoRAs to keep active (default: 1)")
    ap.add_argument("--lora-dtype", default=None, help="LoRA dtype for vLLM (auto/float16/bfloat16)")
    return ap.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _sanitize_model_name(model_id: Optional[str]) -> str:
    if not model_id:
        return "unknown_model"
    name = model_id.rstrip("/").split("/")[-1]
    return name.replace(":", "_")


def _default_output_root(cfg: Dict[str, Any], model_id: Optional[str], prompt_format: str) -> str:
    experiment = cfg.get("benchmark", "experiment")
    model_name = _sanitize_model_name(model_id)
    return osp.join("results", f"llm_{prompt_format}", model_name, experiment)


def _prompt_for_config() -> str:
    config_root = osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))), "configs", "llm"
    )
    candidates = []
    if osp.isdir(config_root):
        candidates = sorted([p for p in os.listdir(config_root) if p.endswith(".json")])
    if candidates:
        print("Available configs in configs/llm:")
        for name in candidates:
            print(f"  - {name}")
    raw = input("Enter path to config JSON: ").strip()
    if not raw:
        raise SystemExit("No config provided. Exiting.")
    if not osp.isabs(raw):
        candidate = osp.join(config_root, raw)
        if osp.exists(candidate):
            raw = candidate
    if not osp.exists(raw):
        raise SystemExit(f"Config not found: {raw}")
    return raw


def merge_args_with_config(args_cli: argparse.Namespace, cfg: Dict[str, Any]) -> SimpleNamespace:
    graph_key = args_cli.graph or cfg["graph"]["default"]
    graph_root = args_cli.graph_root or cfg["graph"]["options"].get(graph_key)
    if graph_root is None:
        options = ", ".join(sorted(cfg["graph"]["options"].keys()))
        raise ValueError(f"Graph key {graph_key} not found in config. Options: {options}")

    if args_cli.split:
        split = args_cli.split
    elif "all" in cfg["data"]["splits"]:
        split = "all"
    elif "val" in cfg["data"]["splits"]:
        split = "val"
    else:
        split = list(cfg["data"]["splits"].keys())[0]

    model_cfg = cfg.get("model", {})
    gen_cfg = cfg.get("generation", {})
    model_id = args_cli.model or model_cfg.get("id")
    prompt_format = args_cli.prompt_format

    out_root = args_cli.out_root or cfg.get("output", {}).get("output_path")
    if not out_root:
        out_root = _default_output_root(cfg, model_id, prompt_format)

    structured_output_cfg = cfg.get("structured_output") or {}

    lora_cfg = cfg.get("lora", {}) if isinstance(cfg.get("lora", {}), dict) else {}
    lora_path = args_cli.lora_path or model_cfg.get("lora_path") or lora_cfg.get("path")
    lora_name = args_cli.lora_name or model_cfg.get("lora_name") or lora_cfg.get("name")
    lora_id = args_cli.lora_id or model_cfg.get("lora_id") or lora_cfg.get("id")
    max_lora_rank = (
        args_cli.max_lora_rank
        or model_cfg.get("max_lora_rank")
        or lora_cfg.get("max_lora_rank")
        or lora_cfg.get("r")
    )
    max_loras = args_cli.max_loras or model_cfg.get("max_loras") or lora_cfg.get("max_loras")
    lora_dtype = args_cli.lora_dtype or model_cfg.get("lora_dtype") or lora_cfg.get("dtype")

    use_edges = cfg["graph"].get("use_edges", True) if args_cli.use_edges is None else args_cli.use_edges
    use_attributes = (
        cfg["graph"].get("use_attributes", True) if args_cli.use_attributes is None else args_cli.use_attributes
    )
    if use_attributes:
        use_colors = cfg["graph"].get("use_colors", True)
        use_size = cfg["graph"].get("use_size", True)
    else:
        use_colors = False
        use_size = False
    neighbor_cap = (
        cfg["graph"].get("neighbor_cap", 5) if args_cli.neighbor_cap is None else args_cli.neighbor_cap
    )

    merged = SimpleNamespace(
        graph=graph_key,
        graph_root=graph_root,
        split=split,
        scene=args_cli.scene,
        max_queries=args_cli.max_queries,
        out_root=out_root,
        prompt_format=prompt_format,
        model=model_id,
        dtype=args_cli.dtype or model_cfg.get("dtype", "auto"),
        quantization=args_cli.quantization or model_cfg.get("quantization", "none"),
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        max_model_len=model_cfg.get("max_model_len"),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
        swap_space=model_cfg.get("swap_space_gb", 4),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        enable_prefix_caching=(
            model_cfg.get("enable_prefix_caching", False)
            if args_cli.enable_prefix_caching is None
            else args_cli.enable_prefix_caching
        ),
        enforce_eager=model_cfg.get("enforce_eager", False),
        llm_log_stats=args_cli.llm_log_stats,
        batch_size=gen_cfg.get("batch_size", 8),
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 256),
        top_p=gen_cfg.get("top_p", 0.9),
        n=gen_cfg.get("n", 1),
        use_edges=use_edges,
        use_attributes=use_attributes,
        use_colors=use_colors,
        use_size=use_size,
        neighbor_cap=neighbor_cap,
        max_objects=cfg["graph"].get("max_objects", 120),
        save_per_query=cfg["evaluation"].get("save_per_query", False),
        compute_metrics=cfg.get("evaluation", {}).get(
            "compute_metrics",
            cfg.get("evaluation", {}).get("compute_iou", True),
        ),
        structured_outputs=_build_structured_outputs_params(cfg),
        allow_rationale=bool(structured_output_cfg.get("allow_rationale", False)),
        lora_path=lora_path,
        lora_name=lora_name,
        lora_id=lora_id,
        max_lora_rank=max_lora_rank,
        max_loras=max_loras,
        lora_dtype=lora_dtype,
        cfg=cfg,
    )
    return merged


def main():
    args_cli = parse_args()
    if not args_cli.config:
        args_cli.config = _prompt_for_config()
    cfg = load_config(args_cli.config)
    args = merge_args_with_config(args_cli, cfg)

    benchmark = cfg["benchmark"].lower()
    if benchmark == "scanrefer":
        benchmarks.run_scanrefer(cfg, args)
    elif benchmark == "reason3d":
        benchmarks.run_reason3d(cfg, args)
    elif benchmark == "scan2cap":
        benchmarks.run_scan2cap(cfg, args)
    elif benchmark == "scanqa":
        benchmarks.run_scanqa(cfg, args)
    elif benchmark == "sqa3d":
        benchmarks.run_sqa3d(cfg, args)
    elif benchmark in ("surprise3d", "multi3drefer"):
        benchmarks.run_surprise3d(cfg, args)
    else:
        raise ValueError(f"Unsupported benchmark {benchmark}")


if __name__ == "__main__":
    main()
