#!/usr/bin/env python3
"""Batch-compare Whisper (Voclaude test mode) vs Qwen3-ASR on local recordings."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"}


@dataclass
class FileResult:
    file_name: str
    source_path: str
    converted_wav: str
    duration_seconds: float | None
    whisper_seconds: float | None
    whisper_text: str
    whisper_chars: int
    whisper_repetition_ratio: float
    qwen_seconds: float | None
    qwen_text: str
    qwen_chars: int
    qwen_language: str | None
    qwen_repetition_ratio: float
    has_cyrillic_whisper: bool
    has_latin_whisper: bool
    has_cyrillic_qwen: bool
    has_latin_qwen: bool
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compare Whisper and Qwen ASR.")
    parser.add_argument(
        "--input-dir",
        default=r"C:\Users\matvei\Documents\Sound Recordings",
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Qwen model id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional file limit for quick runs (0 = all files).",
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=640,
        help="Max new tokens passed to Qwen transcription.",
    )
    return parser.parse_args()


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    stderr_to_null: bool = False,
) -> subprocess.CompletedProcess[str]:
    stdout_target: Any = subprocess.PIPE
    stderr_target: Any = subprocess.DEVNULL if stderr_to_null else subprocess.PIPE
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=stdout_target,
        stderr=stderr_target,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def ffprobe_duration_seconds(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def convert_to_wav_16k_mono(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    proc = run_command(cmd)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg conversion failed: {detail}")


def extract_whisper_transcript(output: str) -> str:
    marker = ">>> TRANSCRIPTION:"
    if marker in output:
        return output.split(marker, 1)[1].strip()

    text_match = re.search(r"Text:\s*(.+)", output)
    if text_match:
        return text_match.group(1).strip()

    return output.strip()


def repetition_ratio(text: str, ngram: int = 5) -> float:
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    if len(tokens) < ngram:
        return 0.0

    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def score_quality(result: FileResult) -> str:
    # Very simple heuristic summary, not a WER proxy.
    # We prefer lower repetition and non-empty output.
    if result.error:
        return "error"
    if not result.whisper_text and not result.qwen_text:
        return "both-empty"
    if not result.whisper_text:
        return "qwen-better"
    if not result.qwen_text:
        return "whisper-better"

    whisper_rep = result.whisper_repetition_ratio
    qwen_rep = result.qwen_repetition_ratio
    if abs(whisper_rep - qwen_rep) > 0.08:
        return "whisper-better" if whisper_rep < qwen_rep else "qwen-better"

    # If repetition is similar, use longer coherent output as tie-breaker.
    if abs(result.whisper_chars - result.qwen_chars) > 40:
        return "whisper-better" if result.whisper_chars > result.qwen_chars else "qwen-better"

    return "tie"


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    whisper_exe = root / "target" / "debug" / "voclaude.exe"
    if not whisper_exe.exists():
        print(f"Whisper executable not found: {whisper_exe}", file=sys.stderr)
        return 2

    qwen_python = root / "tools" / "qwen3_asr_smoke" / ".venv" / "Scripts" / "python.exe"
    qwen_script = root / "tools" / "qwen3_asr_smoke" / "transcribe.py"
    if not qwen_python.exists():
        print(f"Qwen venv python not found: {qwen_python}", file=sys.stderr)
        return 2
    if not qwen_script.exists():
        print(f"Qwen script not found: {qwen_script}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "tools" / "qwen3_asr_smoke" / "runs" / stamp
    converted_dir = run_dir / "converted"
    out_dir = run_dir / "outputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda path: path.name.lower(),
    )
    if args.limit > 0:
        files = files[: args.limit]

    print(f"Run dir: {run_dir}")
    print(f"Found {len(files)} audio files.")

    results: list[FileResult] = []
    started = time.perf_counter()

    for idx, src in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {src.name}")
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)
        wav_path = converted_dir / f"{safe_stem}.wav"
        duration = ffprobe_duration_seconds(src)

        record = FileResult(
            file_name=src.name,
            source_path=str(src),
            converted_wav=str(wav_path),
            duration_seconds=duration,
            whisper_seconds=None,
            whisper_text="",
            whisper_chars=0,
            whisper_repetition_ratio=0.0,
            qwen_seconds=None,
            qwen_text="",
            qwen_chars=0,
            qwen_language=None,
            qwen_repetition_ratio=0.0,
            has_cyrillic_whisper=False,
            has_latin_whisper=False,
            has_cyrillic_qwen=False,
            has_latin_qwen=False,
            error=None,
        )

        try:
            convert_to_wav_16k_mono(src, wav_path)

            t0 = time.perf_counter()
            whisper_proc = run_command(
                [str(whisper_exe), "--test", str(wav_path)],
                stderr_to_null=True,
            )
            record.whisper_seconds = time.perf_counter() - t0
            whisper_text = extract_whisper_transcript(whisper_proc.stdout or "")
            record.whisper_text = whisper_text
            record.whisper_chars = len(whisper_text)
            record.whisper_repetition_ratio = repetition_ratio(whisper_text)
            record.has_cyrillic_whisper = has_cyrillic(whisper_text)
            record.has_latin_whisper = has_latin(whisper_text)

            t1 = time.perf_counter()
            qwen_proc = run_command(
                [
                    str(qwen_python),
                    str(qwen_script),
                    str(wav_path),
                    "--model",
                    args.qwen_model,
                    "--max-new-tokens",
                    str(args.qwen_max_new_tokens),
                    "--json",
                ],
            )
            record.qwen_seconds = time.perf_counter() - t1

            if qwen_proc.returncode != 0:
                qwen_err = (qwen_proc.stderr or "").strip()
                raise RuntimeError(f"Qwen failed: {qwen_err}")

            payload = json.loads((qwen_proc.stdout or "").strip())
            result0 = payload.get("results", [{}])[0]
            qwen_text = str(result0.get("text", ""))
            record.qwen_text = qwen_text
            record.qwen_chars = len(qwen_text)
            lang = result0.get("language")
            record.qwen_language = str(lang) if lang is not None else None
            record.qwen_repetition_ratio = repetition_ratio(qwen_text)
            record.has_cyrillic_qwen = has_cyrillic(qwen_text)
            record.has_latin_qwen = has_latin(qwen_text)

        except Exception as exc:
            record.error = str(exc)

        results.append(record)

        single_json = out_dir / f"{safe_stem}.json"
        single_json.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    elapsed = time.perf_counter() - started
    summary_json = run_dir / "summary.json"
    summary_json.write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_name",
                "duration_seconds",
                "whisper_seconds",
                "qwen_seconds",
                "whisper_chars",
                "qwen_chars",
                "qwen_language",
                "whisper_rep",
                "qwen_rep",
                "quality_heuristic",
                "error",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.file_name,
                    item.duration_seconds,
                    item.whisper_seconds,
                    item.qwen_seconds,
                    item.whisper_chars,
                    item.qwen_chars,
                    item.qwen_language,
                    item.whisper_repetition_ratio,
                    item.qwen_repetition_ratio,
                    score_quality(item),
                    item.error,
                ]
            )

    md_path = run_dir / "report.md"
    lines = [
        "# Batch ASR Comparison",
        "",
        f"- Input dir: `{input_dir}`",
        f"- Qwen model: `{args.qwen_model}`",
        f"- Qwen max_new_tokens: `{args.qwen_max_new_tokens}`",
        f"- Files processed: `{len(results)}`",
        f"- Total wall time: `{elapsed:.1f}s`",
        "",
        "| File | Dur(s) | Whisper(s) | Qwen(s) | Qwen Lang | Heuristic |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in results:
        lines.append(
            "| {file} | {dur:.1f} | {ws:.1f} | {qs:.1f} | {lang} | {heur} |".format(
                file=item.file_name,
                dur=item.duration_seconds or 0.0,
                ws=item.whisper_seconds or 0.0,
                qs=item.qwen_seconds or 0.0,
                lang=item.qwen_language or "",
                heur=score_quality(item),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Heuristic is only a rough quality signal (repetition/emptiness), not WER.")
    lines.append("- Full per-file transcripts are in `outputs/*.json`.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {md_path}")
    print(f"Per-file outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
