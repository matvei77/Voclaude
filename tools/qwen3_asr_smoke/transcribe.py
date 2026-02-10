#!/usr/bin/env python3
"""
Minimal smoke runner for Qwen3-ASR.

This script is intentionally self-contained so we can quickly validate:
1) model loading
2) Russian/English transcription behavior
3) rough latency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def configure_stdout_utf8() -> None:
    # Windows terminals can default to cp1252; force UTF-8 for multilingual output.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-ASR transcription on one or more audio files."
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="Path(s) to local audio files (wav/flac/mp3/etc).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Hugging Face model id. Example: Qwen/Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help='Language name (e.g. "English", "Russian") or "auto".',
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device map: "auto", "cpu", "cuda:0", etc.',
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=640,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable timestamp output using the forced aligner.",
    )
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model id (used only with --timestamps).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser.parse_args()


def resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str, torch: Any) -> Any:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32

    # Auto policy:
    # - GPU: prefer bf16 when available, else fp16
    # - CPU: fp32 for compatibility
    if device.startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_result_text(result: Any) -> str:
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def get_result_language(result: Any) -> str | None:
    if hasattr(result, "language"):
        return str(result.language)
    return None


def get_result_timestamps(result: Any) -> Any:
    if hasattr(result, "time_stamps"):
        return result.time_stamps
    if hasattr(result, "timestamps"):
        return result.timestamps
    return None


def validate_audio_inputs(paths: list[str]) -> list[str]:
    validated: list[str] = []
    missing: list[str] = []

    for path in paths:
        if Path(path).exists():
            validated.append(path)
        else:
            missing.append(path)

    if missing:
        raise FileNotFoundError(f"Missing audio file(s): {', '.join(missing)}")
    return validated


def main() -> int:
    configure_stdout_utf8()
    args = parse_args()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:
        print(
            "Failed to import dependencies. "
            "Install requirements from tools/qwen3_asr_smoke/requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    try:
        audio_paths = validate_audio_inputs(args.audio)
    except Exception as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(args.dtype, device, torch)
    language = None if args.language.lower() == "auto" else args.language

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.timestamps:
        model_kwargs["forced_aligner"] = args.forced_aligner
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype if device.startswith("cuda") else torch.float32,
            "device_map": device,
        }

    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(args.model, **model_kwargs)
    load_s = time.perf_counter() - t0

    transcribe_input: str | list[str]
    if len(audio_paths) == 1:
        transcribe_input = audio_paths[0]
    else:
        transcribe_input = audio_paths

    t1 = time.perf_counter()
    outputs = model.transcribe(
        audio=transcribe_input,
        language=language,
        return_time_stamps=args.timestamps,
    )
    infer_s = time.perf_counter() - t1

    # Normalize to list for output formatting.
    if not isinstance(outputs, list):
        outputs = [outputs]

    records: list[dict[str, Any]] = []
    for i, result in enumerate(outputs):
        record: dict[str, Any] = {
            "index": i,
            "audio": audio_paths[i] if i < len(audio_paths) else None,
            "language": get_result_language(result),
            "text": get_result_text(result),
        }
        if args.timestamps:
            record["timestamps"] = get_result_timestamps(result)
        records.append(record)

    if args.json:
        payload = {
            "model": args.model,
            "device": device,
            "dtype": str(dtype),
            "load_seconds": round(load_s, 3),
            "inference_seconds": round(infer_s, 3),
            "results": records,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Model: {args.model}")
        print(f"Device: {device} | DType: {dtype}")
        print(f"Load: {load_s:.2f}s | Inference: {infer_s:.2f}s")
        print("")
        for record in records:
            print(f"[{record['index']}] {record['audio']}")
            if record["language"]:
                print(f"Language: {record['language']}")
            print(f"Text: {record['text']}")
            if args.timestamps:
                print(f"Timestamps: {record.get('timestamps')}")
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
