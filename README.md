# go-whisper-ct2

Go bindings to [CTranslate2](https://github.com/OpenNMT/CTranslate2) for high-quality Whisper speech-to-text inference — **without Python**.

This library provides the same transcription quality as [faster-whisper](https://github.com/SYSTRAN/faster-whisper) through a clean Go API and CLI tool. It uses the same CTranslate2 inference engine and model format, just accessed directly from Go/C++ instead of Python.

## Features

- High-quality Whisper transcription via CTranslate2
- No Python dependency — pure Go + C++ implementation
- Support for all Whisper model sizes (tiny, base, small, medium, large-v3)
- Multiple audio formats (WAV, MP3, FLAC, OGG, AIFF, AU)
- Automatic language detection (99 languages supported)
- Translation to English from any supported language
- Multiple output formats (text, JSON, SRT, VTT)
- Quantization support (int8, float16, float32)
- Thread-safe concurrent transcription

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [CLI Usage](#cli-usage)
- [Go Library Usage](#go-library-usage)
- [Compute Types & Performance](#compute-types--performance)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

## Requirements

### System Dependencies

**Arch Linux:**
```bash
sudo pacman -S cmake base-devel pkgconf libsndfile libsamplerate openblas
```

**Ubuntu/Debian:**
```bash
sudo apt install cmake build-essential pkg-config \
    libsndfile1-dev libsamplerate0-dev libopenblas-dev
```

**Fedora:**
```bash
sudo dnf install cmake gcc-c++ pkg-config libsndfile-devel libsamplerate-devel openblas-devel
```

**macOS:**
```bash
brew install cmake pkg-config libsndfile libsamplerate openblas
```

### CTranslate2

CTranslate2 must be installed on your system. Build from source:

```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
mkdir build && cd build

# For CPU-only (recommended for most users):
cmake .. -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release

# For CUDA GPU support:
# cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
sudo make install
sudo ldconfig
```

Verify installation:
```bash
pkg-config --libs ctranslate2
# Should output: -lctranslate2
```

## Installation

### Building from Source

```bash
git clone https://github.com/xPrimeTime/go-whisper-ct2.git
cd go-whisper-ct2
make
```

This builds:
- C++ shared library (`csrc/build/libwhisper_ct2.so`)
- Go package (`pkg/whisper`)
- CLI binary (`bin/whisper-ct2`)

### Installing System-Wide

```bash
# Install C++ library (requires sudo)
sudo make install-cpp

# Install CLI to your Go bin directory
make install
```

### As a Go Library

```bash
go get github.com/xPrimeTime/go-whisper-ct2/pkg/whisper
```

**Note:** The C++ library must be built and installed for the Go package to work.

## Model Setup

Whisper models must be in CTranslate2 format. You have two options:

### Option 1: Download Pre-Converted Models (Recommended)

Pre-converted models are available on Hugging Face from Systran:

```bash
# Install huggingface-hub CLI (one-time)
pip install huggingface-hub

# Download a model
huggingface-cli download Systran/faster-whisper-small --local-dir whisper-small-ct2

# Available models:
# - Systran/faster-whisper-tiny      (~150 MB)
# - Systran/faster-whisper-base      (~300 MB)
# - Systran/faster-whisper-small     (~1 GB)
# - Systran/faster-whisper-medium    (~3 GB)
# - Systran/faster-whisper-large-v3  (~6 GB)
```

**Note:** Pre-converted models use `float16` precision. On CPUs without float16 support (most CPUs), CTranslate2 will automatically convert to `float32` at runtime. You'll see a warning message, but this is normal and doesn't affect transcription quality.

### Option 2: Convert Models Yourself

For custom quantization or specific model versions:

```bash
# Install conversion tools (one-time)
pip install ctranslate2 transformers[torch]

# Convert with int8 quantization (fastest on CPU, smallest size)
ct2-transformers-converter --model openai/whisper-small \
    --output_dir whisper-small-ct2-int8 \
    --quantization int8

# Convert with float32 (no runtime conversion warning)
ct2-transformers-converter --model openai/whisper-small \
    --output_dir whisper-small-ct2-fp32 \
    --quantization float32
```

Available OpenAI models:
- `openai/whisper-tiny` - Fastest, least accurate
- `openai/whisper-base` - Good balance for real-time
- `openai/whisper-small` - Recommended for most uses
- `openai/whisper-medium` - Higher accuracy
- `openai/whisper-large-v3` - Best accuracy, slowest

## CLI Usage

### Basic Usage

```bash
# Transcribe an audio file
whisper-ct2 -model ./whisper-small-ct2 audio.wav

# Specify language (faster than auto-detection)
whisper-ct2 -model ./whisper-small-ct2 -language en audio.wav

# Translate foreign language to English
whisper-ct2 -model ./whisper-small-ct2 -task translate german_audio.wav
```

### Output Formats

```bash
# Plain text (default)
whisper-ct2 -model ./model audio.wav

# JSON with metadata
whisper-ct2 -model ./model -output json audio.wav

# SRT subtitles
whisper-ct2 -model ./model -output srt audio.wav > subtitles.srt

# WebVTT subtitles
whisper-ct2 -model ./model -output vtt audio.wav > subtitles.vtt
```

### CLI Options

```
Required:
  -model string       Path to CTranslate2 model directory

Audio Options:
  -language string    Language code (e.g., "en", "es", "zh") or "auto" (default "auto")
  -task string        "transcribe" or "translate" to English (default "transcribe")

Output Options:
  -output string      Output format: text, json, srt, vtt (default "text")

Performance Options:
  -beam-size int      Beam search width, higher = more accurate but slower (default 5)
  -compute-type string  Compute precision: int8, float16, float32, default
  -threads int        CPU threads per operation, 0 = auto (default 0)

Other:
  -verbose            Show progress and timing information
  -version            Print version and exit
```

### Examples

```bash
# Fast transcription with int8
whisper-ct2 -model ./whisper-small-ct2 -compute-type int8 audio.wav

# High accuracy with larger beam
whisper-ct2 -model ./whisper-large-v3-ct2 -beam-size 10 audio.wav

# Process multiple files
for f in *.wav; do
    whisper-ct2 -model ./model -output srt "$f" > "${f%.wav}.srt"
done
```

## Go Library Usage

### Basic Transcription

```go
package main

import (
    "fmt"
    "log"

    "github.com/xPrimeTime/go-whisper-ct2/pkg/whisper"
)

func main() {
    // Load model with default config
    model, err := whisper.LoadModel("./whisper-small-ct2", whisper.DefaultModelConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()

    // Transcribe audio file
    result, err := model.TranscribeFile("audio.wav")
    if err != nil {
        log.Fatal(err)
    }

    // Print transcription
    fmt.Println(result.Text)
}
```

### With Options

```go
result, err := model.TranscribeFile("audio.wav",
    whisper.WithLanguage("en"),           // Skip auto-detection
    whisper.WithTask("transcribe"),       // or "translate"
    whisper.WithBeamSize(5),              // Beam search width
    whisper.WithScores(true),             // Include confidence scores
)
```

### Language Detection

```go
// Detect language without full transcription
probs, err := model.DetectLanguage("audio.wav")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Detected: %s (%.1f%% confidence)\n",
    probs[0].Language,
    probs[0].Probability * 100)
```

### Working with Segments

```go
result, err := model.TranscribeFile("audio.wav")
if err != nil {
    log.Fatal(err)
}

// Access individual segments with timestamps
for _, seg := range result.Segments {
    fmt.Printf("[%v -> %v] %s\n", seg.Start, seg.End, seg.Text)
}

// Generate subtitle formats
fmt.Print(result.SRT())  // SRT format
fmt.Print(result.VTT())  // WebVTT format
```

### Model Configuration

```go
config := whisper.ModelConfig{
    Device:       "cpu",       // "cpu" or "cuda"
    ComputeType:  "int8",      // "int8", "float16", "float32", "default"
    InterThreads: 1,           // Parallel batch processing
    IntraThreads: 4,           // Threads per operation (0 = auto)
}

model, err := whisper.LoadModel("./model", config)
```

### Raw PCM Audio

```go
// Transcribe raw PCM samples (16kHz, mono, float32)
samples := []float32{...} // Your audio data
result, err := model.TranscribePCM(samples, whisper.WithLanguage("en"))
```

## Compute Types & Performance

### Understanding Compute Types

| Type | Size | Speed | Accuracy | Best For |
|------|------|-------|----------|----------|
| `int8` | Smallest | Fastest | Slightly lower | CPU inference, real-time |
| `float16` | Medium | Fast | Full | GPU inference |
| `float32` | Largest | Baseline | Full | CPU without float16 support |

### CPU Users

Most CPUs don't have native float16 support. When using float16 models, you'll see:
```
[warning] The compute type inferred from the saved model is float16, but the target device
or backend do not support efficient float16 computation. The model weights have been
automatically converted to use the float32 compute type instead.
```

**This is normal and harmless.** The transcription works correctly. To avoid the warning:

1. Use `int8` quantized models (recommended - faster too!)
2. Convert models with `--quantization float32`
3. Set compute type explicitly: `-compute-type float32`

### Recommended Setup by Use Case

| Use Case | Model | Compute Type |
|----------|-------|--------------|
| Real-time/streaming | whisper-tiny or base | int8 |
| General transcription | whisper-small | int8 |
| High accuracy | whisper-medium or large-v3 | int8 or float32 |
| GPU inference | Any | float16 |

## Troubleshooting

### Library not found

```
error while loading shared libraries: libwhisper_ct2.so: cannot open shared object file
```

**Solutions:**

```bash
# Option 1: Set library path temporarily
export LD_LIBRARY_PATH=/path/to/go-whisper-ct2/csrc/build:$LD_LIBRARY_PATH

# Option 2: Install system-wide
sudo make install-cpp
sudo ldconfig

# Option 3: Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export LD_LIBRARY_PATH=/path/to/go-whisper-ct2/csrc/build:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### CTranslate2 not found during build

```
Could not find a package configuration file provided by "ctranslate2"
```

**Solutions:**

```bash
# Check if CTranslate2 is installed
pkg-config --libs ctranslate2

# If not found, set PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Or reinstall CTranslate2
cd CTranslate2/build && sudo make install && sudo ldconfig
```

### Model loading errors

```
whisper: failed to load model: ...
```

**Check:**
1. Model directory contains `model.bin` and `config.json`
2. Path is correct (use absolute path if unsure)
3. Model was converted for CTranslate2 (not raw PyTorch)

### Audio loading errors

```
whisper: failed to load audio: ...
```

**Supported formats:** WAV, MP3, FLAC, OGG, AIFF, AU

**Check:**
1. File exists and is readable
2. Audio is not corrupted
3. libsndfile is installed: `pkg-config --libs sndfile`

### Slow transcription

**Tips:**
1. Use `int8` compute type: `-compute-type int8`
2. Use smaller model (tiny or base for real-time)
3. Specify language instead of auto-detect: `-language en`
4. Reduce beam size: `-beam-size 1`

## Project Structure

```
go-whisper-ct2/
├── csrc/                       # C++ implementation
│   ├── include/
│   │   └── whisper_ct2.h      # Public C API
│   ├── src/
│   │   ├── whisper_ct2.cpp    # Main implementation
│   │   ├── audio_processor.*  # Audio loading & preprocessing
│   │   ├── mel_filters.*      # Mel spectrogram filterbank
│   │   └── stft.*             # Short-time Fourier transform
│   ├── third_party/
│   │   └── pocketfft/         # FFT library (header-only)
│   └── CMakeLists.txt
├── pkg/whisper/                # Go package
│   ├── whisper.go             # Main API & cgo bindings
│   ├── model.go               # Model loading
│   ├── transcribe.go          # Transcription functions
│   ├── options.go             # Functional options
│   ├── result.go              # Result types & formatting
│   └── errors.go              # Error handling
├── cmd/whisper-ct2/           # CLI application
│   └── main.go
├── Makefile                   # Build orchestration
├── go.mod
├── README.md
├── DESIGN.md                  # Technical design document
└── LICENSE
```

## How It Works

1. **Audio Loading**: libsndfile loads audio files, libsamplerate resamples to 16kHz
2. **Preprocessing**: STFT computed with PocketFFT, converted to log mel spectrogram
3. **Inference**: CTranslate2 runs the Whisper encoder-decoder model
4. **Decoding**: Beam search generates text tokens with timestamps
5. **Output**: Tokens cleaned (BPE artifacts removed) and formatted

The mel spectrogram computation matches OpenAI's original implementation and faster-whisper exactly, ensuring identical transcription quality.

## Differences from faster-whisper

| Feature | faster-whisper | go-whisper-ct2 |
|---------|---------------|----------------|
| Language | Python | Go + C++ |
| Runtime dependency | Python + packages | None (single binary) |
| Model format | CTranslate2 | CTranslate2 (same) |
| Transcription quality | Reference | Identical |
| Word-level timestamps | Yes | Not yet |
| VAD filtering | Yes | Not yet |
| Streaming | Yes | File-based only |

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference engine
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Reference implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original model
- [PocketFFT](https://github.com/mreineck/pocketfft) - FFT implementation
- [libsndfile](http://www.mega-nerd.com/libsndfile/) - Audio file I/O
- [libsamplerate](http://www.mega-nerd.com/SRC/) - Sample rate conversion
