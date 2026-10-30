# Voclaude

Voice input anywhere. Press a key, speak, press the key again, paste.

Voclaude is a small Windows tray application that turns speech into text on
your own computer. Nothing is sent to a server. It understands 25 European
languages, including English, German, French, Dutch and Russian, and it keeps
up when you switch language in the middle of a sentence. The text is on your
clipboard a fraction of a second after you stop talking, because Voclaude
transcribes while you are still speaking.

- Local speech model: [Parakeet Ultra](https://huggingface.co/handy-computer/parakeet-ultra-gguf) (0.6 billion parameters, 611 MB on disk).
- Runs on any GPU with a Vulkan driver (NVIDIA, AMD, Intel) or on the CPU.
- No Python, no cloud account, no CUDA toolkit.
- Local API, so your scripts and agents can transcribe files with the same model.

## Install (Windows)

1. Open the [Releases](https://github.com/matvei77/Voclaude/releases) page.
2. Download the newest `voclaude-vX.Y.Z-setup.exe` and run it.
3. If Windows shows "Windows protected your PC", click **More info**, then
   **Run anyway**. The installer is not code-signed yet.
4. Keep the "Start Voclaude when I sign in" box checked if you want Voclaude
   to be ready after every login. Click Install.

The installer needs no administrator rights. It installs into
`%LOCALAPPDATA%\Programs\Voclaude` and starts the app. A Voclaude icon appears
in the system tray (the bottom-right corner, next to the clock). Windows 11
often hides new tray icons behind the `^` chevron ("Show hidden icons"). Click
the chevron and drag the Voclaude icon onto the taskbar to keep it visible.

On the first dictation, Voclaude downloads the speech model (about 0.6 GB)
from Hugging Face into `%USERPROFILE%\.cache\huggingface\hub`. This happens
once. It needs an internet connection and takes a few minutes. The first run
on a GPU also compiles the GPU programs once (about 10 s). Later starts load
the model in 1 to 2 s.

Prefer no installer? `voclaude-vX.Y.Z-<hash>.zip` unzips anywhere. Run
`voclaude.exe` from that folder.

Releases before 0.6.0 ship the older Qwen3-ASR engine and the CUDA runtime.
They still work, but they are more than 20 times larger and do not include Parakeet.

### Requirements

| | Minimum |
|---|---|
| Windows | Windows 10 or 11, 64-bit |
| CPU | 64-bit with AVX2 (Intel since 2013, AMD since 2015) |
| GPU (optional) | Any NVIDIA, AMD or Intel GPU with a Vulkan driver and about 1 GB of free video memory |
| Microphone | Any. In Windows Settings, Privacy, Microphone, allow desktop apps. |
| Disk | 0.7 GB for the model and the app |
| Internet | Once, for the model download |

Without a usable GPU, Voclaude uses the CPU. On a desktop CPU this is still
far faster than speech (see [Speed](#speed-and-latency)).

## Use it

1. Press **F4**. The tray icon turns red. Speak.
2. Press **F4** again. The icon returns to normal and the text is on the clipboard.
3. Press **Ctrl+V** in any program to paste.

That is the whole workflow. Some details:

- **Mixed languages.** Voclaude cuts the recording into segments at pauses and
  recognizes each segment in its own language. An English sentence, a Russian
  sentence and another English sentence come out in their own alphabets.
  A short Russian phrase inside an English sentence is also caught: words the
  model was unsure of are decoded again on their own (`parakeet_recheck`).
- **Formatting.** The first letter is capitalized and a space is added at the
  end, so you can paste several dictations in a row. Both can be turned off.
- **Long dictations.** Recordings longer than 10 minutes, or transcripts
  longer than 20,000 characters, are saved as a text file in the transcripts
  folder. In that case the clipboard holds the path to the file, not the text.
- **Nothing gets lost.** Audio is written to disk while you speak and synced
  every 5 s. Every finished segment's text is journaled. If the computer
  crashes or loses power, Voclaude resumes from the last finished segment when
  it starts again. "Recover Last Recording" in the tray menu re-runs a failed
  transcription by hand.
- **Idle footprint.** Inference runs in a separate process that exits after
  60 s without work. While idle, Voclaude uses no video memory and about 20 MB
  of RAM.

### The tray menu

Right-click the Voclaude icon:

| Item | What it does |
|---|---|
| *model and device* | Shows which model is loaded and whether it runs on the GPU or the CPU. |
| Start Recording (F4) / Stop Recording (F4) | Same as the hotkey. |
| History (Ctrl+Shift+H) | Opens the History window: every transcription, searchable, with a Copy button per entry. |
| Open Last Transcript | Opens the newest transcription in Notepad. Opens History when nothing has been dictated yet. |
| Open Transcripts Folder | Opens the folder with saved transcripts. |
| Recover Last Recording | Enabled only when the last recording did not finish. Re-runs the missing part. |
| Edit Settings (config.toml) | Opens the settings file in Notepad. Changes saved within five minutes apply without a restart, except the hotkeys. |
| Open Logs Folder | Opens the log files, useful when reporting a problem. |
| Quit Voclaude | Stops the app. |

### Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `F4` | Start or stop recording |
| `Ctrl+Shift+H` | Show or hide the History window |

Both are configurable (see [Settings](#settings)).

## Languages

Parakeet Ultra recognizes 25 languages and detects the language of each
segment by itself:

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French,
German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish,
Portuguese, Romanian, Russian, Slovak, Slovene, Spanish, Swedish, Ukrainian.

Output is punctuated and cased. The model transcribes what it hears and does
not translate.

### Accuracy

Word error rate on FLEURS read speech, 40 clips per language, human
references, lower is better (measured 2026-10-23, details in
[docs/bench/2026-10-23-model-survey.md](docs/bench/2026-10-23-model-survey.md)):

| Model | ru | en | de | fr | nl | average |
|---|---|---|---|---|---|---|
| **Parakeet Ultra Q6_K** (default) | 5.1 | 4.7 | 4.7 | 5.0 | 7.3 | 5.4 |
| Qwen3-ASR-1.7B (optional engine) | 5.0 | 3.6 | 5.3 | 4.4 | 7.4 | 5.1 |
| *Cloud: Voxtral Small Transcribe 3* | 0.7 | 1.7 | 1.0 | 2.5 | 1.2 | 1.4 |
| *Cloud: Gemini 4 Argon* | 2.1 | 3.3 | 3.2 | 3.0 | 3.5 | 3.0 |

The honest gap: on read speech, cloud models make two to four times fewer
errors. On real dictation with Russian and English mixed, no cloud model
tested kept both languages reliably, and in a blind reading of a 58-minute
mixed meeting, readers preferred Parakeet over Qwen3-ASR-1.7B in 45 segments,
Qwen in 24, and saw no difference in 54.

## Speed and latency

### How the latency works

Voclaude does not wait for you to finish and then transcribe the whole
recording. While you speak, the recorder cuts the audio into segments:

1. A segment is at least 6 s long.
2. After that, the first pause of 0.5 s closes the segment.
3. If no pause comes, the segment is cut at its quietest point before 45 s.

Each closed segment goes to the speech model immediately, while you keep
talking. When you press the stop key, only the last open segment is still
untranscribed. The wait you notice is the time to transcribe that one segment,
plus joining the text and writing the clipboard.

Measured on the development machine (RTX 3060, Vulkan) from the app's own logs
on 2026-10-29, 40 dictations of 1 to 32 segments: the wait from the stop key
to the clipboard was 0.04 to 0.56 s, with one outlier at 1.32 s. A short last
segment finishes in well under 0.1 s. The model loads in 0.8 to 2.1 s, but
that happens while you speak the first words, because the worker starts when
the recording starts.

### Throughput

"Realtime factor" is audio seconds transcribed per second of compute. 100x
means one minute of speech in 0.6 s.

| Hardware | Parakeet Ultra Q6_K | Wait per 20 s segment |
|---|---|---|
| NVIDIA RTX 3060, Vulkan | about 145x on a 58-minute meeting (170 live segments), up to 182x on short segments | under 0.15 s |
| Intel i5-13600K, CPU | about 26x | about 0.8 s |
| Laptop CPU (Intel Core Ultra class) | not measured, estimated 10 to 15x | estimated 1.5 to 2 s |
| Apple Silicon | not measured for Parakeet | |

All CPU work uses the right ggml module for the processor (AVX2, AVX-512 and
so on), picked at start.

The optional Qwen3-ASR-1.7B engine runs at about 29x on the RTX 3060 (CUDA)
and 2.5x on the i5-13600K. Anything above about 1.5x keeps up with speech,
because segments are transcribed while you talk. Numbers are from
[docs/bench/](docs/bench/README.md).

## Settings

Voclaude writes a commented settings file on the first start:

`%APPDATA%\voclaude\Voclaude\config\config.toml`

Open it from the tray menu ("Edit Settings"). The most useful options:

```toml
hotkey = "F4"                   # start or stop recording
history_hotkey = "Ctrl+Shift+H"  # show or hide the History window
add_trailing_space = true
capitalize_first = true
use_gpu = true                  # false = always use the CPU
engine = "parakeet"             # or "qwen" (see below)
parakeet_model = "handy-computer/parakeet-ultra-gguf/parakeet-ultra-Q6_K.gguf"
parakeet_recheck = true         # keep short Russian phrases inside English in Cyrillic
cpu_threads = 0                 # CPU threads for Parakeet, 0 = automatic
idle_unload_seconds = 60        # free the GPU this long after the last dictation
streaming = true                # transcribe segments while recording
segment_max_seconds = 45
segment_pause_seconds = 0.5
api_enabled = true
api_port = 7770
```

See [config.example.toml](config.example.toml) for every option with
comments. Changes saved within five minutes of opening the file from the tray
menu apply without a restart, except the hotkeys. Later edits apply at the next
start.

**Hotkey format.** Modifiers `Ctrl`, `Alt`, `Shift`, `Win` joined with `+`,
then a key: letters, digits, `F1` to `F12`, `Space`, `Enter`, `Tab`, `Escape`,
numpad keys and punctuation. Examples: `F4`, `Win+C`, `Ctrl+Shift+Space`.

**Other Parakeet files.** The same Hugging Face repository has `Q8_0`
(740 MB, same accuracy, 40 percent slower on the CPU) and `Q4_K_M` (486 MB,
same accuracy and speed as Q6_K). A local `.gguf` path also works.

**The Qwen engine.** `engine = "qwen"` switches to
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) through pure
Rust inference (candle). It is a little better on English and worse on mixed
Russian and English, and it is 2 GB. On the GPU it needs the CUDA build
(`voclaude-vX.Y.Z-cuda-setup.exe`, NVIDIA GTX 16xx or RTX 20xx or newer, 6 GB
of video memory, a 2025 or newer driver). On the CPU it runs in every build.

## Local API

While Voclaude runs, it serves an OpenAI-compatible transcription endpoint on
`http://127.0.0.1:7770`. Only programs on the same computer can reach it. The
endpoint uses the same loaded model as dictation, decodes audio and video
files (wav, mp3, flac, ogg, m4a, mp4, mov, mkv, and anything else through
`ffmpeg` when it is installed), and adds every result to the History window.

```
voclaude transcribe meeting.mp4
curl -F file=@clip.wav http://127.0.0.1:7770/v1/audio/transcriptions
```

Any OpenAI Whisper client works when its base URL is
`http://127.0.0.1:7770/v1`. Set `api_token` in the settings to require a
bearer token, or `api_enabled = false` to turn the API off. Full reference:
[docs/API.md](docs/API.md).

## Command line

`voclaude.exe` is also a command-line tool. On Windows it attaches to the
terminal it was started from.

| Command | Purpose |
|---|---|
| `voclaude --version` | Version and git hash |
| `voclaude --validate` | Load the configured model, check the GPU and the microphone, exit |
| `voclaude --list-models` | The default Parakeet file and the Qwen tiers |
| `voclaude --test audio.wav` | Transcribe one file with the configured engine and print the text |
| `voclaude transcribe <file>...` | Send files to the running app's API and print the text |
| `voclaude --bench <dir>` | Benchmark harness (Qwen and Whisper engines only) |

Options for `--validate` and `--test`: `--cpu`, `--gpu`, `--engine parakeet|qwen`,
`--parakeet-model <spec>`, `--threads N`, `--model-tier fast|best`.

## Build from source

### Windows

Prerequisites:

- [Rust](https://rustup.rs) 1.88 or newer.
- Visual Studio Build Tools 2022 with the "Desktop development with C++"
  workload. Cargo needs the MSVC linker, and `fetch-transcribe.ps1` needs
  `dumpbin.exe` and `lib.exe` from the same toolset.
- Git.

Build:

```powershell
git clone https://github.com/matvei77/Voclaude.git
cd Voclaude
.\fetch-transcribe.ps1     # once per shell: downloads the transcribe.cpp runtime, sets TRANSCRIBE_DIR
cargo build --release
.\target\release\voclaude.exe
```

`fetch-transcribe.ps1` downloads the official transcribe.cpp 0.4.1 Windows
runtime (CPU and Vulkan DLLs, checksum verified) into `vendor/` and exports
`TRANSCRIBE_DIR` for the current shell. Run it again in every new shell before
`cargo build`, otherwise Cargo compiles transcribe.cpp from source without
Vulkan.

The default build needs no CUDA toolkit. Parakeet runs on Vulkan or the CPU,
and the Qwen engine runs on the CPU.

**CUDA build** (adds the Qwen engine on NVIDIA GPUs): install the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x, then

```powershell
.\package.ps1 -Cuda          # finds cl.exe through vswhere, builds with --features cuda
```

or `cargo build --release --features cuda` from a Visual Studio developer
shell (nvcc needs `cl.exe` on PATH).

**Packaging:**

```powershell
.\package.ps1                     # dist\voclaude-vX.Y.Z-<hash>.zip
.\package.ps1 -Installer          # also dist\voclaude-vX.Y.Z-setup.exe (needs Inno Setup 6)
.\package.ps1 -Cuda -Installer    # the CUDA variant: -cuda.zip and -cuda-setup.exe
```

The zip holds `voclaude.exe`, the transcribe.cpp DLLs, the MSVC runtime DLLs,
`config.example.toml` and a short README. The CUDA variant adds the CUDA
runtime DLLs, so recipients only need an NVIDIA driver.

### Linux

```bash
# Ubuntu / Debian
sudo apt install build-essential cmake pkg-config libasound2-dev libgtk-3-dev \
    libayatana-appindicator3-dev libxkbcommon-dev libxdo-dev
cargo build --release
./target/release/voclaude
```

On Linux, Cargo compiles transcribe.cpp from source and links it statically.
That build is CPU-only: the Vulkan backend is not enabled in `Cargo.toml` for
non-Windows targets. The Linux build has not been validated with the Parakeet
engine. Under Wayland, global hotkeys may need `GDK_BACKEND=x11`.

### macOS

```bash
cargo build --release --features metal
```

Compiles on Apple Silicon in CI (`.github/workflows/macos-smoke.yml`), where
the Qwen engine was measured on Metal. The tray and hotkey code has not been
run on a Mac, and Parakeet through Metal has not been measured.

### Continuous integration

- `ci.yml` builds and tests the standard Windows build on every push and pull
  request. No CUDA toolkit is involved.
- `release.yml` builds the zip and installer when a `v*` tag is pushed, plus
  the CUDA variant and a macOS build on a best-effort basis.

## Architecture

Two processes. The tray process owns the microphone, the hotkeys, the
clipboard, the History window and the local API. The worker process owns the
speech model. "Unloading" the model means the worker exits, which returns all
GPU and host memory to the system and keeps a crash in the model runtime from
taking the recorder down.

```
tray process (voclaude.exe)                              worker process (voclaude.exe --worker)
┌────────────────────────────────────────────────┐       ┌─────────────────────────────────────┐
│ global hotkey (F4) ──► event loop (app.rs)     │ JSON  │ Engine (inference/mod.rs)           │
│ tray icon + menu (tray.rs)      │              │ lines │   Parakeet: transcribe.cpp / ggml   │
│ History window, egui (ui.rs)    │              │ over  │     Vulkan GPU or per-ISA CPU       │
│ local API :7770 (api.rs)        ▼              │ pipes │   Qwen3-ASR: candle (CUDA or CPU)   │
│ mic ─► CPAL ─► 16 kHz ─► segmenter ────────────┼───────┼─► transcribe one segment ─► text    │
│          (audio/)        │      journal        │       │                                     │
│                          ▼     (session.rs)    │◄──────┤ exits after idle_unload_seconds     │
│ clipboard ◄── join segments + format           │       └─────────────────────────────────────┘
│ history.json (history.rs)                      │
└────────────────────────────────────────────────┘
```

Data on disk (all under `%APPDATA%\voclaude\Voclaude`): `config\config.toml`,
`data\history.json`, `data\session.json` (the crash-recovery journal),
`data\transcripts\`, `data\logs\`, and the recordings themselves as raw 16 kHz
`data\audio-<timestamp>.f32` files, kept for the last 10 dictations
(`audio_retention_count`).

The module guide, data-flow diagrams and conventions are in
[docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md). The assessment of what to build
next is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Troubleshooting

**"Windows protected your PC" when running the installer.** The installer is
not code-signed. Click "More info", then "Run anyway".

**Nothing happens when I press F4.** Another program may own the F4 hotkey.
Change `hotkey` in the settings, for example to `Ctrl+Shift+Space`. Hover the
tray icon: the tooltip shows the last error, such as a missing microphone.

**No microphone found.** Check the default input device in Windows sound
settings, and that desktop apps may use the microphone (Settings, Privacy,
Microphone). Voclaude re-opens the device on every recording, so plugging in a
microphone later works without a restart.

**The first dictation takes long.** The model downloads once (0.6 GB). Watch
the tray tooltip. The transcription of that first dictation starts after the
download.

**It runs on the CPU although I have a GPU.** Run `voclaude --validate` in a
terminal. It prints the device transcribe.cpp picked. Update the GPU driver if
no Vulkan device is listed. `use_gpu = true` must be set in the settings.

**Russian came out in Latin letters.** Make sure `parakeet_recheck = true`
(the default) and that the Russian part is spoken with a short pause around it
when it is longer than a few words, so it gets its own segment.

**Something else.** Tray menu, "Open Logs Folder". The newest
`voclaude.log.YYYY-MM-DD` has the timeline of every event.

## Credits and licenses

- Voclaude is MIT licensed.
- Speech model: [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)
  by Moondream, a post-trained variant of NVIDIA's
  [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  both licensed CC-BY-4.0. The GGUF conversion is
  [handy-computer/parakeet-ultra-gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf).
- Inference runtime: [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
  (MIT), built on ggml (MIT). Its license files ship in the `licenses` folder
  of every package.
- Optional engine: [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
  (Apache-2.0) through [candle](https://github.com/huggingface/candle).
