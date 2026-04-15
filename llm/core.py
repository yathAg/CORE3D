#!/usr/bin/env python3
"""Core helpers for vLLM inference (model init, generation, parsing, metrics logging)."""

import atexit
import contextlib
import copy
import csv
import json
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

try:
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover
    LoRARequest = None


# NVTX shim: fall back to no-op contexts if NVTX or CUDA is unavailable.
try:
    import torch.cuda.nvtx as _torch_nvtx

    class _SafeNVTX:
        def range(self, *args, **kwargs):
            try:
                return _torch_nvtx.range(*args, **kwargs)
            except Exception:
                return contextlib.nullcontext()

    nvtx = _SafeNVTX()
except Exception:  # pragma: no cover
    class _NoopNVTX:
        def range(self, *args, **kwargs):
            return contextlib.nullcontext()

    nvtx = _NoopNVTX()


# Lazily created vLLM engine and tokenizer.
_LLM = None
_TOKENIZER = None
_LORA_REQUEST = None


def _resolve_dtype(dtype_str: str) -> str:
    if dtype_str not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype {dtype_str} for vLLM")
    return dtype_str


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _extract_prediction(parsed: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    if "prediction" in parsed:
        return parsed.get("prediction"), "prediction"
    return None, None


def _coerce_prediction_text(pred: Any) -> str:
    if pred is None:
        return ""
    if isinstance(pred, list):
        return ", ".join(str(p) for p in pred)
    return str(pred)


def _build_structured_outputs_params(cfg: Dict[str, Any]) -> Optional[StructuredOutputsParams]:
    so_cfg = cfg.get("structured_output") or {}
    if not so_cfg.get("enabled", False):
        return None
    allow_rationale = bool(so_cfg.get("allow_rationale", False))
    constraint_keys = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
    params = {k: so_cfg.get(k) for k in constraint_keys if so_cfg.get(k) is not None}
    if "json" in params and isinstance(params["json"], dict):
        schema = copy.deepcopy(params["json"])
        props = schema.get("properties")
        if isinstance(props, dict):
            if allow_rationale:
                props.setdefault("rationale", {"type": "string"})
            else:
                props.pop("rationale", None)
            schema["properties"] = props
        required = schema.get("required")
        if isinstance(required, list):
            if allow_rationale and "rationale" not in required:
                pass
            else:
                schema["required"] = [r for r in required if r != "rationale"]
        params["json"] = schema
    if not params:
        raise ValueError("structured_output enabled but no constraint specified in config.")
    for key in (
        "disable_fallback",
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
    ):
        if key in so_cfg:
            params[key] = so_cfg[key]
    return StructuredOutputsParams(**params)


def _build_script_args(args) -> Dict[str, Any]:
    script_args = dict(vars(args))
    if "structured_outputs" in script_args:
        cfg = script_args.get("cfg") or {}
        script_args["structured_outputs"] = cfg.get("structured_output")
    return script_args


def _build_lora_request(args) -> Optional["LoRARequest"]:
    lora_path = getattr(args, "lora_path", None)
    if not lora_path:
        return None
    if LoRARequest is None:
        raise RuntimeError("LoRA support is unavailable in this vLLM build.")
    if not osp.exists(lora_path):
        raise SystemExit(f"LoRA adapter path not found: {lora_path}")
    lora_name = getattr(args, "lora_name", None) or osp.basename(lora_path.rstrip("/"))
    lora_id = int(getattr(args, "lora_id", None) or 1)
    return LoRARequest(lora_name, lora_id, lora_path)


def _get_lora_request(args) -> Optional["LoRARequest"]:
    global _LORA_REQUEST
    if _LORA_REQUEST is None:
        _LORA_REQUEST = _build_lora_request(args)
    return _LORA_REQUEST


def _get_llm_and_tokenizer(args):
    global _LLM, _TOKENIZER
    if _LLM is None or _TOKENIZER is None:
        dtype = _resolve_dtype(args.dtype)
        quantization = None if args.quantization == "none" else args.quantization
        lora_request = _get_lora_request(args)
        lora_kwargs = {}
        if lora_request is not None:
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": int(getattr(args, "max_loras", None) or 1),
                "max_lora_rank": int(getattr(args, "max_lora_rank", None) or 16),
                "lora_dtype": getattr(args, "lora_dtype", None) or "auto",
            }
        with nvtx.range("init_vllm_engine"):
            _LLM = LLLM = LLM(
                model=args.model,
                tokenizer=args.model,
                dtype=dtype,
                trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                swap_space=args.swap_space,
                enable_prefix_caching=args.enable_prefix_caching,
                quantization=quantization,
                enforce_eager=args.enforce_eager,
                disable_log_stats=not bool(getattr(args, "compute_metrics", True)),
                **lora_kwargs,
            )
        with nvtx.range("get_tokenizer"):
            _TOKENIZER = _LLM.get_tokenizer()
        tok = _TOKENIZER
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        lora_msg = ""
        if lora_request is not None:
            lora_msg = (
                f" lora_path={lora_request.lora_path} "
                f"lora_id={lora_request.lora_int_id} "
                f"max_lora_rank={lora_kwargs.get('max_lora_rank')}"
            )
        print(
            f"[model] loaded {args.model} dtype={dtype} tp={args.tensor_parallel_size} "
            f"quantization={quantization or 'none'} prefix_cache={args.enable_prefix_caching} "
            f"max_model_len={args.max_model_len} enforce_eager={args.enforce_eager}{lora_msg}"
        )
    return _LLM, _TOKENIZER


def _shutdown_llm():
    global _LLM, _LORA_REQUEST
    if _LLM is not None:
        try:
            _LLM.shutdown()
        except Exception:
            pass
        _LLM = None
    _LORA_REQUEST = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


atexit.register(_shutdown_llm)


def call_model_batch_vllm(
    entries: List[Dict[str, Any]],
    args,
    token_acc: Optional[Dict[str, int]] = None,
    stats_totals: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    llm, tokenizer = _get_llm_and_tokenizer(args)
    structured_outputs = getattr(args, "structured_outputs", None)
    allow_rationale = bool(getattr(args, "allow_rationale", False))
    lora_request = _get_lora_request(args)

    with nvtx.range("build_chat_strings"):
        chat_strings = []
        for e in entries:
            messages = [{"role": "user", "content": e["prompt"]}]
            chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_strings.append(chat_str)

    do_sample = args.temperature > 0
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature if do_sample else 0.0,
        top_p=0.9 if do_sample else 1.0,
        n=1,
        structured_outputs=structured_outputs,
    )

    gen_kwargs = {"use_tqdm": args.llm_log_stats}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    with torch.inference_mode():
        with nvtx.range("vllm_generate"):
            outputs = llm.generate(
                chat_strings,
                sampling_params,
                **gen_kwargs,
            )

    results: List[Dict[str, Any]] = []
    for out, entry in zip(outputs, entries):
        text = out.outputs[0].text
        if token_acc is not None:
            token_acc["input"] = token_acc.get("input", 0) + len(out.prompt_token_ids)
            token_acc["output"] = token_acc.get("output", 0) + len(out.outputs[0].token_ids)
            token_acc["samples"] = token_acc.get("samples", 0) + 1
        if stats_totals is not None:
            _accumulate_request_stats(stats_totals, getattr(out, "metrics", None))

        parsed = _extract_json(text)
        pred_val = None
        if parsed is not None:
            pred_val, _ = _extract_prediction(parsed)
        if parsed is None or pred_val is None:
            results.append(
                {
                    "prediction": None,
                    "parse_fail": True,
                    "_raw": text,
                }
            )
        else:
            result_obj = {
                "prediction": pred_val,
                "parse_fail": False,
                "_raw": text,
            }
            if allow_rationale:
                result_obj["rationale"] = parsed.get("rationale", "")
            results.append(result_obj)

    return results


def _fmt_csv(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _init_stats_totals() -> Dict[str, float]:
    return {
        "samples": 0,
        "queue_time_sec": 0.0,
        "prefill_time_sec": 0.0,
        "decode_time_sec": 0.0,
        "inference_time_sec": 0.0,
        "ttft_sec": 0.0,
    }


def _accumulate_request_stats(stats_totals: Dict[str, float], metrics: Any) -> None:
    if not metrics:
        return
    required = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
    if not all(hasattr(metrics, key) for key in required):
        return
    try:
        queued_ts = float(getattr(metrics, "queued_ts"))
        scheduled_ts = float(getattr(metrics, "scheduled_ts"))
        first_token_ts = float(getattr(metrics, "first_token_ts"))
        last_token_ts = float(getattr(metrics, "last_token_ts"))
    except Exception:
        return
    if queued_ts <= 0 or scheduled_ts <= 0 or first_token_ts <= 0 or last_token_ts <= 0:
        return
    queued_time = scheduled_ts - queued_ts
    prefill_time = first_token_ts - scheduled_ts
    decode_time = last_token_ts - first_token_ts
    inference_time = last_token_ts - scheduled_ts
    if min(queued_time, prefill_time, decode_time, inference_time) < 0:
        return
    try:
        ttft = float(getattr(metrics, "first_token_latency"))
    except Exception:
        return
    if ttft < 0:
        return
    stats_totals["queue_time_sec"] += queued_time
    stats_totals["prefill_time_sec"] += prefill_time
    stats_totals["decode_time_sec"] += decode_time
    stats_totals["inference_time_sec"] += inference_time
    stats_totals["ttft_sec"] += ttft
    stats_totals["samples"] += 1


def _collect_vllm_engine_metrics() -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    llm = _LLM
    if llm is None:
        return metrics
    engine = getattr(llm, "llm_engine", None)
    if engine is None or not getattr(engine, "log_stats", False):
        return metrics
    try:
        raw_metrics = engine.get_metrics()
    except Exception:
        return metrics
    kv_values: List[float] = []
    prefix_hits = 0
    prefix_queries = 0
    for item in raw_metrics:
        name = getattr(item, "name", "")
        if name == "vllm:kv_cache_usage_perc" and hasattr(item, "value"):
            try:
                kv_values.append(float(item.value) * 100.0)
            except Exception:
                pass
        elif name == "vllm:prefix_cache_hits" and hasattr(item, "value"):
            try:
                prefix_hits += int(item.value)
            except Exception:
                pass
        elif name == "vllm:prefix_cache_queries" and hasattr(item, "value"):
            try:
                prefix_queries += int(item.value)
            except Exception:
                pass
    if kv_values:
        metrics["kv_cache_usage_perc"] = sum(kv_values) / len(kv_values)
    if prefix_queries > 0:
        metrics["prefix_cache_hit_rate"] = prefix_hits / prefix_queries
    return metrics


def _write_run_metrics_csv(
    out_root: str,
    split: str,
    token_totals: Dict[str, int],
    timing_totals: Dict[str, float],
    stats_totals: Optional[Dict[str, float]] = None,
    engine_metrics: Optional[Dict[str, float]] = None,
) -> str:
    total_samples = token_totals.get("samples", 0)
    total_time = timing_totals.get("time_sec", 0.0)
    avg_latency_ms = (total_time / timing_totals["samples"] * 1000) if timing_totals.get("samples") else None
    avg_input_tokens = (token_totals.get("input", 0) / total_samples) if total_samples else None
    avg_output_tokens = (token_totals.get("output", 0) / total_samples) if total_samples else None
    prompt_tokens_per_sec = (token_totals.get("input", 0) / total_time) if total_time > 0 else None
    gen_tokens_per_sec = (token_totals.get("output", 0) / total_time) if total_time > 0 else None
    stats_samples = stats_totals.get("samples", 0) if stats_totals else 0
    avg_queue_time_ms = (
        (stats_totals["queue_time_sec"] / stats_samples * 1000) if stats_samples else None
    )
    avg_prefill_time_ms = (
        (stats_totals["prefill_time_sec"] / stats_samples * 1000) if stats_samples else None
    )
    avg_decode_time_ms = (
        (stats_totals["decode_time_sec"] / stats_samples * 1000) if stats_samples else None
    )
    avg_inference_time_ms = (
        (stats_totals["inference_time_sec"] / stats_samples * 1000) if stats_samples else None
    )
    avg_ttft_ms = (
        (stats_totals["ttft_sec"] / stats_samples * 1000) if stats_samples else None
    )
    kv_cache_usage_perc = engine_metrics.get("kv_cache_usage_perc") if engine_metrics else None
    prefix_cache_hit_rate = engine_metrics.get("prefix_cache_hit_rate") if engine_metrics else None

    csv_path = osp.join(out_root, f"run_metrics_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "samples",
                "total_time_sec",
                "avg_latency_ms",
                "total_input_tokens",
                "total_output_tokens",
                "avg_input_tokens",
                "avg_output_tokens",
                "prompt_tokens_per_sec",
                "gen_tokens_per_sec",
                "avg_queue_time_ms",
                "avg_prefill_time_ms",
                "avg_decode_time_ms",
                "avg_inference_time_ms",
                "avg_time_to_first_token_ms",
                "kv_cache_usage_perc",
                "prefix_cache_hit_rate",
            ]
        )
        writer.writerow(
            [
                split,
                total_samples,
                _fmt_csv(total_time),
                _fmt_csv(avg_latency_ms),
                token_totals.get("input", 0),
                token_totals.get("output", 0),
                _fmt_csv(avg_input_tokens),
                _fmt_csv(avg_output_tokens),
                _fmt_csv(prompt_tokens_per_sec),
                _fmt_csv(gen_tokens_per_sec),
                _fmt_csv(avg_queue_time_ms),
                _fmt_csv(avg_prefill_time_ms),
                _fmt_csv(avg_decode_time_ms),
                _fmt_csv(avg_inference_time_ms),
                _fmt_csv(avg_ttft_ms),
                _fmt_csv(kv_cache_usage_perc),
                _fmt_csv(prefix_cache_hit_rate),
            ]
        )
    return csv_path
