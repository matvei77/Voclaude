# Local API

While Voclaude is running it serves a small HTTP API on `127.0.0.1:7770`
(set `api_port` in `config.toml`; `api_enabled = false` turns it off). Only
processes on the same machine can reach it. Set `api_token = "..."` to require
`Authorization: Bearer <token>`.

Transcriptions use the same loaded model as dictation. Files are decoded
in-process (wav, mp3, flac, ogg/vorbis, m4a/mp4/mov with AAC or ALAC, mkv);
anything else falls back to `ffmpeg` if it is on `PATH`. Long files are cut at
pauses into chunks of up to 90 s and queued behind whatever dictation segment
is in flight, so dictating while a batch job runs still works.

## Command line

```
voclaude transcribe meeting.mp4
voclaude transcribe --json a.wav b.m4a
voclaude transcribe --verbose-json lecture.mkv   # per-chunk timestamps
```

Prints the text (or JSON) to stdout. Exit code 1 if any file failed, 2 for
usage errors. Requires the tray app to be running.

## HTTP

`POST /v1/audio/transcriptions` accepts three body styles:

- **OpenAI-compatible multipart** (`file` part, optional `response_format`):

  ```
  curl -F file=@clip.wav http://127.0.0.1:7770/v1/audio/transcriptions
  ```

- **Local path as JSON** (no upload; the file is read from disk):

  ```
  curl -H "Content-Type: application/json" \
       -d '{"path": "C:\\Videos\\talk.mp4", "response_format": "text"}' \
       http://127.0.0.1:7770/v1/audio/transcriptions
  ```

- **Raw audio body** (any container), with an optional `X-Filename` header
  for the extension hint.

`response_format` is `json` (default, `{"text": ..., "duration": ...}`),
`text` (plain text), or `verbose_json` (adds `segments` with start/end seconds
per chunk). It can also be given as a query parameter, as can `path`.

Other routes: `GET /health` (model, device, version) and `GET /v1/models`.

Errors come back as `{"error": {"message": "..."}}` with 400 (bad request or
path not found), 401 (token), 413 (upload over 2 GiB), 415 (undecodable), 500
(inference failed), 503 (app shutting down), or 504 (chunk timed out).

Every completed request is added to the History window like a dictation.

## Using it from an agent

Any OpenAI Whisper client works by pointing its base URL at
`http://127.0.0.1:7770/v1`. From a shell, `voclaude transcribe <file>` is the
shortest path: it resolves the path, calls the endpoint, and prints the text.
