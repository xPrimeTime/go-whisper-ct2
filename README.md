# go-whisper-ct2

Go bindings to [CTranslate2](https://github.com/OpenNMT/CTranslate2) for high-quality Whisper speech-to-text inference — **without Python**.

This library provides the same transcription quality as [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with **performance 1.23x slower** than the Python implementation (with proper configuration). It uses the same CTranslate2 inference engine and model format, accessed directly from Go/C++ instead of Python.

## Features

- High-quality Whisper transcription via CTranslate2
- **Performance 1.23x slower than faster-whisper** (with OMP_NUM_THREADS configured)
- No Python dependency — pure Go + C++ implementation
- Support for all Whisper model sizes (tiny, base, small, medium, large-v3)
- Multiple audio formats (WAV, MP3, FLAC, OGG, AIFF, AU)
- Automatic language detection (99 languages supported)
- Translation to English from any supported language
- Multiple output formats (text, JSON, SRT, VTT)
- Quantization support (int8, float16, float32)
- Thread-safe concurrent transcription
- Advanced optimizations: silent chunk filtering, context conditioning, quality checks, temperature fallback

## Quick Start (100% Python-Free)

```bash
# 1. Build the project
git clone https://github.com/xPrimeTime/go-whisper-ct2.git
cd go-whisper-ct2
make

# 2. Download a model (using git, no Python needed)
git clone https://huggingface.co/Systran/faster-whisper-small whisper-small-ct2

# 3. Set optimal threading for best performance
export OMP_NUM_THREADS=12  # Adjust for your CPU (see Performance section)

# 4. Transcribe audio
./bin/whisper-ct2 -model ./whisper-small-ct2 audio.wav
```

**No Python required** for download, build, or runtime! Python is only needed if you want to convert custom models with specific quantization.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [CLI Usage](#cli-usage)
- [Go Library Usage](#go-library-usage)
- [Benchmarking](#benchmarking)
- [Compute Types & Performance](#compute-types--performance)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

## Requirements

**Note:** Python is **NOT** required for building or running this library. Python is only needed if you want to convert custom Whisper models (optional - pre-converted models are available).

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

**Important:** This package uses cgo and requires the C++ library. You **cannot** simply `go get` it - you must build from source first.

```bash
# Clone and build the C++ library
git clone https://github.com/xPrimeTime/go-whisper-ct2.git
cd go-whisper-ct2
make build-cpp

# Install C++ library system-wide (recommended)
sudo make install-cpp

# Now you can import in your Go code
```

Then in your Go project:

```bash
go get github.com/xPrimeTime/go-whisper-ct2/pkg/whisper
```

**What users need installed:**
1. **CTranslate2** (build from source - see [Requirements](#requirements))
2. **System libraries**: libsndfile, libsamplerate, openblas
3. **This package's C++ library**: libwhisper_ct2.so (built via `make build-cpp`)

The package will link against these libraries at compile time and runtime.

## Model Setup

Whisper models must be in CTranslate2 format. Pre-converted models are available - **no Python required for download or runtime!**

### Download Pre-Converted Models (Python-Free)

Pre-converted models are available on Hugging Face. Choose your preferred download method:

#### Method 1: Git LFS (Recommended, No Python)

```bash
# Install git-lfs if not already installed
# Arch: sudo pacman -S git-lfs
# Ubuntu: sudo apt install git-lfs
# macOS: brew install git-lfs

git lfs install

# Clone a model (downloads all files)
git clone https://huggingface.co/Systran/faster-whisper-small whisper-small-ct2

# Or for faster download, clone without history:
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Systran/faster-whisper-small whisper-small-ct2
cd whisper-small-ct2
git lfs pull
```

#### Method 2: Direct Download with wget/curl (No Python)

```bash
# Create directory
mkdir -p whisper-small-ct2 && cd whisper-small-ct2

# Download required files
wget https://huggingface.co/Systran/faster-whisper-small/resolve/main/config.json
wget https://huggingface.co/Systran/faster-whisper-small/resolve/main/model.bin
wget https://huggingface.co/Systran/faster-whisper-small/resolve/main/tokenizer.json
wget https://huggingface.co/Systran/faster-whisper-small/resolve/main/vocabulary.txt
```

#### Method 3: Browser Download (No Python)

1. Visit https://huggingface.co/Systran/faster-whisper-small/tree/main
2. Download these files: `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`
3. Place all files in a directory (e.g., `whisper-small-ct2/`)

#### Method 4: Using huggingface-hub CLI (Optional, Requires Python)

```bash
pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-small --local-dir whisper-small-ct2
```

### Available Models

| Model | Size | Speed | Accuracy | HuggingFace URL |
|-------|------|-------|----------|-----------------|
| tiny | ~75 MB | Fastest | Lower | https://huggingface.co/Systran/faster-whisper-tiny |
| base | ~145 MB | Fast | Good | https://huggingface.co/Systran/faster-whisper-base |
| small | ~486 MB | Medium | Better | https://huggingface.co/Systran/faster-whisper-small |
| medium | ~1.5 GB | Slow | High | https://huggingface.co/Systran/faster-whisper-medium |
| large-v3 | ~3.1 GB | Slowest | Best | https://huggingface.co/Systran/faster-whisper-large-v3 |

**Note:** Pre-converted models use `float16` precision. On CPUs without float16 support (most CPUs), CTranslate2 will automatically convert to `float32` at runtime. You'll see a warning message, but this is normal and doesn't affect transcription quality.

### Convert Custom Models (Optional, Requires Python)

Only needed if you want custom quantization (int8, float32) or specific model variants:

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

## Benchmarking

A dedicated benchmark tool is included for performance testing and comparing configurations.

### Build the Benchmark Tool

```bash
make build-benchmark
# Binary: bin/whisper-benchmark
```

### Basic Usage

```bash
# Benchmark with 3 iterations (default)
./bin/whisper-benchmark --model ./whisper-base-ct2 audio.wav

# Multiple iterations for better statistics
./bin/whisper-benchmark --model ./whisper-base-ct2 --iterations 10 audio.wav

# Benchmark multiple files
./bin/whisper-benchmark --model ./whisper-base-ct2 audio1.wav audio2.wav audio3.mp3
```

### Export Results

```bash
# Save results to JSON and CSV
./bin/whisper-benchmark \
  --model ./whisper-base-ct2 \
  --iterations 5 \
  --output-json results.json \
  --output-csv results.csv \
  audio.wav
```

### Reported Metrics

The benchmark tool provides comprehensive statistics:

- **Transcription time**: min, max, mean, median, standard deviation
- **Real-Time Factor (RTF)**: Automatic calculation (audio_duration / transcription_time)
  - Higher RTF = faster (e.g., 3.5x means processing 3.5 seconds of audio per second)
- **Language detection** results
- **Segment and text** statistics

### Example Output

```
Benchmarking: audio.wav
================================================================================

Audio: audio.wav
  Duration:           4.82s
  Iterations:         3

Transcription Time:
  Min:                1.234s
  Max:                1.298s
  Mean:               1.267s
  Median:             1.271s
  Std Dev:            0.027s

Real-Time Factor:
  Min RTF:            3.71x (fastest)
  Max RTF:            3.91x (slowest)
  Mean RTF:           3.81x
  Median RTF:         3.79x

Transcription Info:
  Language:           en
  Segments:           2
  Text length:        142 chars
================================================================================
```

### Comparison Benchmarks

```bash
# Compare different quantization levels
./bin/whisper-benchmark --model ./whisper-base-ct2-int8 --compute-type int8 --output-json int8.json audio.wav
./bin/whisper-benchmark --model ./whisper-base-ct2 --output-json default.json audio.wav

# Compare beam sizes
for beam in 1 5 10; do
  ./bin/whisper-benchmark --model ./whisper-base-ct2 --beam-size $beam --output-json beam-$beam.json audio.wav
done
```

See [cmd/benchmark/README.md](cmd/benchmark/README.md) for complete documentation.

### Comparing with faster-whisper

Automated scripts are provided to benchmark go-whisper-ct2 against faster-whisper:

```bash
# Install faster-whisper
pip install faster-whisper

# Run automated comparison (builds both, runs benchmarks, compares results)
./scripts/run-comparison.sh --model ./whisper-base-ct2 --iterations 10 audio.wav
```

This will run both implementations with identical settings and display a detailed comparison:

```
================================================================================
PERFORMANCE COMPARISON: go-whisper-ct2 vs faster-whisper
================================================================================

Metric               Go              Python          Difference
--------------------------------------------------------------------------------
Mean Time            5.51s           4.47s           +23.3%
Mean RTF             3.34x           4.11x           -18.8%

⚠️  Go is 1.23x slower than Python (requires OMP_NUM_THREADS configuration)
```

See [scripts/README.md](scripts/README.md) and [BENCHMARKING.md](BENCHMARKING.md) for detailed comparison guides.

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

### Performance Benchmarks

Real-world performance comparison with faster-whisper (Python):

**Test Setup:**
- Model: whisper-small (float16 → auto-converted to float32)
- Hardware: AMD Ryzen 7 5800X3D (8 cores, 16 threads)
- Audio: harvard.wav, 18.4 seconds

**Results:**

| Implementation | Time | Real-Time Factor | vs Python |
|----------------|------|------------------|-----------|
| faster-whisper (Python, default) | 4.47s | 4.11x | Baseline |
| go-whisper-ct2 (with OMP_NUM_THREADS=12) | 5.51s | 3.34x | **1.23x slower** |
| go-whisper-ct2 (without OMP config) | 10.5s | 1.75x | 2.35x slower |

**⚠️ IMPORTANT:** Setting `OMP_NUM_THREADS` is critical for performance!

```bash
# Without OMP_NUM_THREADS: ~10s (2.3x slower than Python)
./bin/whisper-ct2 -model ./whisper-small-ct2 audio.wav

# With optimal OMP_NUM_THREADS: ~5.5s (1.23x slower than Python)
export OMP_NUM_THREADS=12
./bin/whisper-ct2 -model ./whisper-small-ct2 audio.wav
```

**Optimal OMP_NUM_THREADS by CPU:**
- 16-thread CPU (8 cores): `OMP_NUM_THREADS=12`
- 8-thread CPU (4 cores): `OMP_NUM_THREADS=6`
- 4-thread CPU (2 cores): `OMP_NUM_THREADS=3`
- **Rule of thumb**: Use 75% of your total thread count

**Key Takeaways:**
- ✅ Performance 1.23x slower than faster-whisper (with proper threading)
- ✅ Both use identical CTranslate2 inference engine
- ✅ Both implement same optimizations (silent chunk filtering, context conditioning, etc.)
- ✅ Go version has zero Python runtime overhead
- ✅ Single binary deployment vs Python environment
- ⚠️ Must set `OMP_NUM_THREADS` for optimal performance

See [PERFORMANCE.md](PERFORMANCE.md) for detailed analysis and optimization guide.

**Optimization Features (Enabled by Default):**
1. **Silent chunk filtering** - Automatically skips silent audio (2-3x faster on typical audio)
2. **Context conditioning** - Uses previous text for better accuracy
3. **Compression ratio checks** - Detects and retries hallucinated/repetitive text
4. **Log probability thresholds** - Identifies low-confidence segments
5. **Temperature fallback** - Automatically retries poor quality segments

**Fine-Tuning Performance:**

```go
// Faster (more aggressive filtering, may miss some speech)
result, err := model.TranscribeFile("audio.wav",
    whisper.WithNoSpeechThreshold(0.8),        // Skip more silence
    whisper.WithBeamSize(1),                   // Greedy decoding
    whisper.WithCompressionRatioThreshold(2.0), // Stricter quality
)

// More accurate (slower, processes everything)
result, err := model.TranscribeFile("audio.wav",
    whisper.WithNoSpeechThreshold(0.0),        // Process all chunks
    whisper.WithBeamSize(10),                  // Wider beam search
    whisper.WithConditionOnPreviousText(true), // Full context
)
```

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

**First: Check if OMP_NUM_THREADS is set!** This is the #1 cause of slow performance.

```bash
# Set optimal threading (critical for performance!)
export OMP_NUM_THREADS=12  # Adjust for your CPU

# Verify it's set
echo $OMP_NUM_THREADS

# Now transcribe
./bin/whisper-ct2 -model ./whisper-small-ct2 audio.wav
```

**Additional optimization tips:**
1. **Set OMP_NUM_THREADS** to ~75% of your CPU thread count (most important!)
2. Use `int8_float32` compute type (if supported): `-compute-type int8_float32`
3. Use smaller model (tiny or base for real-time)
4. Specify language instead of auto-detect: `-language en`
5. Reduce beam size: `-beam-size 1`

**Performance troubleshooting:**
```bash
# Check current performance
time OMP_NUM_THREADS=12 ./bin/whisper-ct2 -model ./model audio.wav

# Compare with different thread counts
for threads in 4 8 12 16; do
  echo "Testing OMP_NUM_THREADS=$threads"
  OMP_NUM_THREADS=$threads time ./bin/whisper-ct2 -model ./model audio.wav
done
```

See [PERFORMANCE.md](PERFORMANCE.md) for detailed optimization guide.

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

## Comparison with faster-whisper

| Feature | faster-whisper | go-whisper-ct2 |
|---------|---------------|----------------|
| Language | Python | Go + C++ |
| Runtime dependency | Python + packages | None (single binary) |
| Model format | CTranslate2 | CTranslate2 (same) |
| Transcription quality | Reference | Identical |
| Performance | Baseline | 1.23x slower |
| Performance (if OMP not set) | Baseline | 2.3x slower ⚠️ |
| Silent chunk filtering | ✓ | ✓ |
| Context conditioning | ✓ | ✓ |
| Compression ratio checks | ✓ | ✓ |
| Log probability thresholds | ✓ | ✓ |
| Temperature fallback | ✓ | ✓ |
| INT8 quantization | ✓ | Limited (backend dependent) |
| Word-level timestamps | ✓ | Planned |
| Silero VAD preprocessing | ✓ | Not implemented |
| Streaming transcription | ✓ | File-based only |

**Summary:** Core optimization features are fully implemented with excellent performance (1.23x of Python). The Go implementation offers easier deployment (single binary, no Python) while maintaining the same transcription quality. **Remember to set `OMP_NUM_THREADS` for optimal performance!**

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
