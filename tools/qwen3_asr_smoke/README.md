# Qwen3-ASR Smoke Test

This folder is a controlled Python-first validation path for Qwen3-ASR before Rust production integration.

Goals:
- verify the model loads on your machine
- verify Russian and English transcription quality quickly
- capture basic latency and output behavior

## Scope

This is intentionally not production code. It is an experiment harness that stays isolated under `tools/`.

## Prerequisites

- Python 3.12 is recommended (3.13 may fail depending on wheel availability for `torch` / `qwen-asr`)
- NVIDIA GPU optional (CPU works, but is much slower)

## Setup

```powershell
cd tools/qwen3_asr_smoke
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## Run

Single file, auto language:

```powershell
python transcribe.py C:\path\to\sample.wav
```

Force Russian:

```powershell
python transcribe.py C:\path\to\sample_ru.wav --language Russian
```

Force English:

```powershell
python transcribe.py C:\path\to\sample_en.wav --language English
```

Bigger model:

```powershell
python transcribe.py C:\path\to\sample.wav --model Qwen/Qwen3-ASR-1.7B
```

Long file stability (manual chunking, single model load):

```powershell
python transcribe.py C:\path\to\long.wav --model Qwen/Qwen3-ASR-1.7B --chunk-seconds 60 --max-new-tokens 2048
```

JSON output:

```powershell
python transcribe.py C:\path\to\sample.wav --json
```

With timestamps (loads forced aligner):

```powershell
python transcribe.py C:\path\to\sample.wav --timestamps
```

Quality-oriented profile for bilingual files:

```powershell
python transcribe.py C:\path\to\sample.wav `
  --model Qwen/Qwen3-ASR-1.7B `
  --dtype bfloat16 `
  --chunk-seconds 60 `
  --max-new-tokens 2048 `
  --json
```

## Notes

- Default model in this script is `Qwen/Qwen3-ASR-0.6B` for faster first validation.
- Qwen3-ASR can return timestamps (`--timestamps`) but does not perform speaker diarization (no speaker labels like `SPEAKER_00`).
- After smoke validation, the next step is to define a stable Rust backend contract and bridge from the Rust app.
