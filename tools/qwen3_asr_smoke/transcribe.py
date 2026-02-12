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
        nargs="*",
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
        "--context",
        default="",
        help="Optional ASR context prompt passed to the model.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Manually split each audio file into fixed-size chunks (seconds). 0 disables.",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=0.0,
        help="Optional overlap between consecutive manual chunks (seconds).",
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
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as persistent server, reading JSON requests from stdin.",
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


def split_waveform_into_chunks(
    wav: Any,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[Any, float]]:
    if chunk_seconds <= 0:
        return [(wav, 0.0)]

    chunk_samples = int(round(chunk_seconds * sample_rate))
    overlap_samples = int(round(overlap_seconds * sample_rate))
    if chunk_samples <= 0:
        raise ValueError("--chunk-seconds must be > 0")
    if overlap_samples < 0:
        raise ValueError("--chunk-overlap-seconds must be >= 0")
    step = chunk_samples - overlap_samples
    if step <= 0:
        raise ValueError(
            "--chunk-overlap-seconds must be smaller than --chunk-seconds"
        )

    total = int(getattr(wav, "shape")[0])
    if total <= chunk_samples:
        return [(wav, 0.0)]

    chunks: list[tuple[Any, float]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_samples)
        chunks.append((wav[start:end], start / float(sample_rate)))
        if end >= total:
            break
        start += step
    return chunks


def merge_languages(langs: list[str | None]) -> str:
    merged: list[str] = []
    prev = ""
    for value in langs:
        if not value:
            continue
        for part in str(value).split(","):
            lang = part.strip()
            if not lang:
                continue
            if lang == prev:
                continue
            merged.append(lang)
            prev = lang
    return ",".join(merged)


def timestamp_items_with_offset(raw_timestamps: Any, offset_sec: float) -> list[dict[str, Any]]:
    if raw_timestamps is None:
        return []

    if hasattr(raw_timestamps, "items"):
        items = list(raw_timestamps.items)
    elif isinstance(raw_timestamps, list):
        items = raw_timestamps
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        text = None
        start_time = None
        end_time = None
        if isinstance(item, dict):
            text = item.get("text")
            start_time = item.get("start_time")
            end_time = item.get("end_time")
        else:
            text = getattr(item, "text", None)
            start_time = getattr(item, "start_time", None)
            end_time = getattr(item, "end_time", None)

        if start_time is not None:
            start_time = round(float(start_time) + offset_sec, 3)
        if end_time is not None:
            end_time = round(float(end_time) + offset_sec, 3)

        normalized.append(
            {
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return normalized


def server_main(args) -> int:
    """Persistent server mode: load model once, process requests from stdin."""
    configure_stdout_utf8()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audio_input
    except Exception as exc:
        error = {"status": "error", "error": f"Failed to import: {exc}"}
        print(json.dumps(error), flush=True)
        return 2

    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(args.dtype, device, torch)

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
        "max_new_tokens": args.max_new_tokens,
    }

    try:
        t0 = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(args.model, **model_kwargs)
        load_s = time.perf_counter() - t0
    except Exception as exc:
        error = {"status": "error", "error": f"Model load failed: {exc}"}
        print(json.dumps(error), flush=True)
        return 2

    ready = {
        "status": "ready",
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "load_seconds": round(load_s, 3),
    }
    print(json.dumps(ready), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"status": "error", "error": f"Invalid JSON: {e}"}
            print(json.dumps(resp), flush=True)
            continue

        if req.get("command") == "shutdown":
            break

        audio_path = req.get("audio_path")
        if not audio_path:
            resp = {"status": "error", "error": "Missing audio_path"}
            print(json.dumps(resp), flush=True)
            continue

        try:
            language = req.get("language")
            chunk_secs = req.get("chunk_seconds", 0.0)
            chunk_overlap = req.get("chunk_overlap_seconds", 0.0)

            t1 = time.perf_counter()

            if chunk_secs and chunk_secs > 0:
                wav = normalize_audio_input(audio_path)
                chunks = split_waveform_into_chunks(
                    wav=wav,
                    sample_rate=SAMPLE_RATE,
                    chunk_seconds=chunk_secs,
                    overlap_seconds=chunk_overlap,
                )
                chunk_audio = [(c, SAMPLE_RATE) for c, _ in chunks]
                outputs = model.transcribe(audio=chunk_audio, language=language)
                text = "\n".join(get_result_text(r) for r in outputs).strip()
            else:
                result = model.transcribe(audio=audio_path, language=language)
                if isinstance(result, list):
                    result = result[0]
                text = get_result_text(result)

            infer_s = time.perf_counter() - t1

            resp = {
                "status": "ok",
                "model": args.model,
                "device": device,
                "load_seconds": 0.0,
                "inference_seconds": round(infer_s, 3),
                "results": [{"text": text}],
            }
        except Exception as e:
            resp = {"status": "error", "error": str(e)}

        print(json.dumps(resp, ensure_ascii=False), flush=True)

    return 0


def main() -> int:
    configure_stdout_utf8()
    args = parse_args()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audio_input
    except Exception as exc:
        print(
            "Failed to import dependencies. "
            "Install requirements from tools/qwen3_asr_smoke/requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    if args.server:
        return server_main(args)

    if not args.audio:
        print("Error: audio file path(s) required (or use --server).", file=sys.stderr)
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

    records: list[dict[str, Any]] = []
    t1 = time.perf_counter()
    if args.chunk_seconds > 0:
        for i, audio_path in enumerate(audio_paths):
            wav = normalize_audio_input(audio_path)
            chunks = split_waveform_into_chunks(
                wav=wav,
                sample_rate=SAMPLE_RATE,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.chunk_overlap_seconds,
            )
            chunk_audio = [(chunk_wav, SAMPLE_RATE) for chunk_wav, _ in chunks]
            chunk_outputs = model.transcribe(
                audio=chunk_audio,
                context=args.context,
                language=language,
                return_time_stamps=args.timestamps,
            )

            chunk_texts: list[str] = []
            chunk_langs: list[str | None] = []
            merged_timestamps: list[dict[str, Any]] = []
            chunk_meta: list[dict[str, Any]] = []
            for j, ((_, offset_sec), chunk_result) in enumerate(zip(chunks, chunk_outputs)):
                chunk_text = get_result_text(chunk_result)
                chunk_lang = get_result_language(chunk_result)
                chunk_texts.append(chunk_text)
                chunk_langs.append(chunk_lang)
                chunk_meta.append(
                    {
                        "chunk_index": j,
                        "offset_seconds": round(offset_sec, 3),
                        "language": chunk_lang,
                        "text": chunk_text,
                    }
                )
                if args.timestamps:
                    merged_timestamps.extend(
                        timestamp_items_with_offset(
                            get_result_timestamps(chunk_result),
                            offset_sec=offset_sec,
                        )
                    )

            record = {
                "index": i,
                "audio": audio_path,
                "language": merge_languages(chunk_langs),
                "text": "\n".join(text for text in chunk_texts if text).strip(),
                "chunk_count": len(chunks),
                "chunks": chunk_meta,
            }
            if args.timestamps:
                record["timestamps"] = merged_timestamps
            records.append(record)
    else:
        transcribe_input: str | list[str]
        if len(audio_paths) == 1:
            transcribe_input = audio_paths[0]
        else:
            transcribe_input = audio_paths

        outputs = model.transcribe(
            audio=transcribe_input,
            context=args.context,
            language=language,
            return_time_stamps=args.timestamps,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]

        for i, result in enumerate(outputs):
            record = {
                "index": i,
                "audio": audio_paths[i] if i < len(audio_paths) else None,
                "language": get_result_language(result),
                "text": get_result_text(result),
            }
            if args.timestamps:
                record["timestamps"] = get_result_timestamps(result)
            records.append(record)

    infer_s = time.perf_counter() - t1

    if args.json:
        payload = {
            "model": args.model,
            "device": device,
            "dtype": str(dtype),
            "context": args.context,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "load_seconds": round(load_s, 3),
            "inference_seconds": round(infer_s, 3),
            "results": records,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Model: {args.model}")
        print(f"Device: {device} | DType: {dtype}")
        if args.context:
            print(f"Context: {args.context}")
        if args.chunk_seconds > 0:
            print(
                f"Manual chunking: {args.chunk_seconds:.1f}s "
                f"(overlap {args.chunk_overlap_seconds:.1f}s)"
            )
        print(f"Load: {load_s:.2f}s | Inference: {infer_s:.2f}s")
        print("")
        for record in records:
            print(f"[{record['index']}] {record['audio']}")
            if record["language"]:
                print(f"Language: {record['language']}")
            if "chunk_count" in record:
                print(f"Chunks: {record['chunk_count']}")
            print(f"Text: {record['text']}")
            if args.timestamps:
                print(f"Timestamps: {record.get('timestamps')}")
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
