# Go-Whisper-CT2 Design Document

## Overview

Go-Whisper-CT2 is a Go binding to CTranslate2 for Whisper speech-to-text inference without Python dependency. It provides the same high-quality transcription as faster-whisper through a clean Go API.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Go Application                          │
│                    (CLI or custom program)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Go Package (pkg/whisper)                     │
│                                                                 │
│  Model          Transcription        Options        Results    │
│  - LoadModel()  - TranscribeFile()  - WithLang()   - Segment   │
│  - Close()      - TranscribePCM()   - WithTask()   - Result    │
│  - IsMulti()    - DetectLanguage()  - WithBeam()              │
└─────────────────────────────────────────────────────────────────┘
                                │ cgo
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   C API (libwhisper_ct2.so)                     │
│                                                                 │
│  whisper_model_load()        whisper_transcribe_file()         │
│  whisper_model_free()        whisper_transcribe_pcm()          │
│  whisper_result_free()       whisper_detect_language_*()       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    C++ Implementation                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │AudioProcessor│  │    STFT      │  │   Mel Filterbank     │ │
│  │  - load()    │  │  - hann()    │  │   - 80/128 bins      │ │
│  │  - resample()│  │  - fft()     │  │   - pre-computed     │ │
│  │  - mel()     │  │              │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CTranslate2                               │
│                                                                 │
│  ctranslate2::models::Whisper                                   │
│  - generate()                                                   │
│  - detect_language()                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
go-whisper-ct2/
├── csrc/                              # C/C++ source code
│   ├── CMakeLists.txt                # CMake build configuration
│   ├── include/
│   │   └── whisper_ct2.h             # Public C API header
│   ├── src/
│   │   ├── whisper_ct2.cpp           # C API implementation
│   │   ├── audio_processor.h         # Audio preprocessing header
│   │   ├── audio_processor.cpp       # Audio loading & mel spectrogram
│   │   ├── stft.h                    # STFT header
│   │   ├── stft.cpp                  # STFT implementation
│   │   ├── mel_filters.h             # Mel filterbank header
│   │   └── mel_filters.cpp           # Pre-computed mel filterbank
│   ├── third_party/
│   │   └── pocketfft/                # Header-only FFT library (BSD)
│   │       └── pocketfft_hdronly.h
│   └── tests/                        # C++ unit tests
│       ├── CMakeLists.txt
│       └── test_audio_processor.cpp
│
├── pkg/whisper/                       # Go package
│   ├── doc.go                        # Package documentation
│   ├── whisper.go                    # cgo bindings
│   ├── model.go                      # Model type and lifecycle
│   ├── transcribe.go                 # Transcription functions
│   ├── options.go                    # Functional options pattern
│   ├── result.go                     # Result types
│   ├── errors.go                     # Error handling
│   └── whisper_test.go               # Unit tests
│
├── cmd/whisper-ct2/                   # CLI application
│   └── main.go
│
├── testdata/                          # Test fixtures
│   └── jfk.wav                       # Test audio file
│
├── scripts/
│   └── download_model.sh             # Model download helper
│
├── go.mod                            # Go module definition
├── go.sum                            # Go dependencies checksum
├── Makefile                          # Top-level build orchestration
├── README.md                         # User documentation
├── DESIGN.md                         # This file
└── LICENSE
```

## C API Design

### Error Handling

The C API uses a dual error reporting mechanism:

1. **Return codes**: Every function returns `whisper_error_t`
2. **Thread-local state**: Detailed error messages stored per-thread

```c
typedef enum {
    WHISPER_OK = 0,
    WHISPER_ERROR_INVALID_ARGUMENT = 1,
    WHISPER_ERROR_MODEL_NOT_FOUND = 2,
    WHISPER_ERROR_MODEL_LOAD_FAILED = 3,
    WHISPER_ERROR_AUDIO_LOAD_FAILED = 4,
    WHISPER_ERROR_TRANSCRIPTION_FAILED = 5,
    WHISPER_ERROR_OUT_OF_MEMORY = 6,
    WHISPER_ERROR_INVALID_MODEL = 7,
    WHISPER_ERROR_NOT_MULTILINGUAL = 8,
    WHISPER_ERROR_UNSUPPORTED_AUDIO_FORMAT = 9,
    WHISPER_ERROR_INTERNAL = 99
} whisper_error_t;

// Thread-local detailed message
const char* whisper_get_last_error_message(void);
```

### Memory Management Contract

| Allocation | Responsibility |
|------------|---------------|
| `whisper_model_t` | Caller frees via `whisper_model_free()` |
| `whisper_result_t` contents | Library allocates, caller frees via `whisper_result_free()` |
| `whisper_language_prob_t*` | Library allocates, caller frees via `whisper_language_probs_free()` |
| String parameters (`const char*`) | Caller owns, must remain valid during call |

### Key Types

```c
// Opaque model handle
typedef struct whisper_model* whisper_model_t;

// Model configuration
typedef struct {
    const char* device;           // "cpu" only
    const char* compute_type;     // "int8", "int16", "float16", "float32", "default"
    int32_t inter_threads;        // Threads for batch parallelization
    int32_t intra_threads;        // Threads within operations (0 = auto)
} whisper_model_config_t;

// Transcription options
typedef struct {
    int32_t beam_size;              // Beam search width (default: 5)
    float patience;                 // Beam search patience (default: 1.0)
    float length_penalty;           // Length penalty (default: 1.0)
    float repetition_penalty;       // Repetition penalty (default: 1.0)
    int32_t no_repeat_ngram_size;   // Prevent n-gram repetition (default: 0)
    int32_t max_length;             // Max tokens per segment (default: 448)
    int32_t sampling_topk;          // Top-k sampling (default: 1)
    float sampling_temperature;     // Sampling temperature (default: 1.0)
    int32_t max_initial_timestamp_index;
    bool suppress_blank;
    const int32_t* suppress_tokens;
    size_t suppress_tokens_count;
    const char* language;           // "en", "auto", etc.
    const char* task;               // "transcribe" or "translate"
    bool return_scores;
    bool return_no_speech_prob;
    bool word_timestamps;
} whisper_transcribe_options_t;

// Transcription segment
typedef struct {
    char* text;
    float start_time;
    float end_time;
    float score;
    float no_speech_prob;
} whisper_segment_t;

// Transcription result
typedef struct {
    char* detected_language;
    float language_probability;
    whisper_segment_t* segments;
    size_t segment_count;
    float duration;
} whisper_result_t;
```

### Key Functions

```c
// Model lifecycle
whisper_error_t whisper_model_load(const char* path, const whisper_model_config_t* config, whisper_model_t* out);
void whisper_model_free(whisper_model_t model);
bool whisper_model_is_multilingual(whisper_model_t model);
int32_t whisper_model_n_mels(whisper_model_t model);

// Transcription
whisper_error_t whisper_transcribe_file(whisper_model_t model, const char* path, const whisper_transcribe_options_t* opts, whisper_result_t* out);
whisper_error_t whisper_transcribe_pcm(whisper_model_t model, const float* samples, size_t count, const whisper_transcribe_options_t* opts, whisper_result_t* out);
void whisper_result_free(whisper_result_t* result);

// Language detection
whisper_error_t whisper_detect_language_file(whisper_model_t model, const char* path, whisper_language_prob_t** out, size_t* count);
void whisper_language_probs_free(whisper_language_prob_t* probs, size_t count);
```

## Audio Processing Pipeline

The audio processing matches faster-whisper exactly for transcription parity:

```
Audio File (.wav, .mp3, .flac, .ogg)
         │
         ▼
┌─────────────────┐
│  libsndfile     │  Load audio file (any format)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  libsamplerate  │  Resample to 16kHz mono
└─────────────────┘
         │
         ▼
┌─────────────────┐
│      STFT       │  n_fft=400, hop=160, Hann window
│   (PocketFFT)   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Mel Filterbank │  80 bins (or 128 for large-v3)
│  (pre-computed) │  Matches librosa exactly
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Log Scale     │  log10(max(x, 1e-10))
│   + Normalize   │  Dynamic range compression
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Pad/Trim       │  30-second chunks (3000 frames)
└─────────────────┘
         │
         ▼
   Mel Spectrogram [1, n_mels, 3000]
```

### STFT Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| FFT size (n_fft) | 400 |
| Hop length | 160 |
| Window | Hann |
| Frequency bins | 201 (n_fft/2 + 1) |

### Mel Spectrogram Parameters

| Parameter | Value |
|-----------|-------|
| Mel bins | 80 (or 128 for large-v3) |
| Min frequency | 0 Hz |
| Max frequency | 8000 Hz |
| Frame rate | 100 fps (16000/160) |
| Chunk length | 30 seconds |
| Frames per chunk | 3000 |

### Log-Mel Computation

```cpp
// Matches faster-whisper exactly
for (auto& row : mel_spec) {
    for (auto& val : row) {
        val = std::log10(std::max(val, 1e-10f));
        max_val = std::max(max_val, val);
    }
}

// Dynamic range compression
float min_val = max_val - 8.0f;
for (auto& row : mel_spec) {
    for (auto& val : row) {
        val = std::max(val, min_val);
        val = (val + 4.0f) / 4.0f;
    }
}
```

## Go Package Design

### Public API

```go
package whisper

// Model represents a loaded Whisper model
type Model struct { /* private */ }

// LoadModel loads a Whisper model from a CTranslate2 model directory
func LoadModel(path string, config ModelConfig) (*Model, error)

// Close releases model resources
func (m *Model) Close() error

// IsMultilingual returns true if the model supports multiple languages
func (m *Model) IsMultilingual() bool

// NumMels returns the number of mel frequency bins
func (m *Model) NumMels() int

// TranscribeFile transcribes audio from a file path
func (m *Model) TranscribeFile(path string, opts ...Option) (*Result, error)

// TranscribePCM transcribes raw PCM audio samples (float32, mono, 16kHz)
func (m *Model) TranscribePCM(samples []float32, opts ...Option) (*Result, error)

// DetectLanguage detects the spoken language
func (m *Model) DetectLanguage(path string) ([]LanguageProb, error)
```

### Functional Options

```go
type Option func(*TranscribeOptions)

func WithLanguage(lang string) Option     // "en", "fr", "auto", etc.
func WithTask(task string) Option         // "transcribe" or "translate"
func WithBeamSize(size int) Option        // Beam search width
func WithWordTimestamps(enabled bool) Option
```

### Result Types

```go
type Segment struct {
    Text         string
    Start        time.Duration
    End          time.Duration
    Score        float32
    NoSpeechProb float32
}

type Result struct {
    Language            string
    LanguageProbability float32
    Segments            []Segment
    Text                string        // Full concatenated text
    Duration            time.Duration
}

type LanguageProb struct {
    Language    string
    Probability float32
}
```

### Error Handling

```go
var (
    ErrInvalidArgument = errors.New("whisper: invalid argument")
    ErrModelNotFound   = errors.New("whisper: model not found")
    ErrModelLoadFailed = errors.New("whisper: failed to load model")
    ErrAudioLoadFailed = errors.New("whisper: failed to load audio")
    ErrTranscribeFailed = errors.New("whisper: transcription failed")
    ErrNotMultilingual = errors.New("whisper: model is not multilingual")
    // ...
)

// Errors include detailed messages from the C layer
type Error struct {
    Code    int
    Message string
    Err     error
}
```

## CLI Tool

```bash
whisper-ct2 [flags] <audio-file>

Flags:
  -m, --model string        Path to CTranslate2 model directory (required)
  -l, --language string     Language code or "auto" (default "auto")
  -t, --task string         "transcribe" or "translate" (default "transcribe")
  -o, --output string       Output format: text, json, srt, vtt (default "text")
  -b, --beam-size int       Beam search width (default 5)
      --compute-type string Compute type: int8, float16, float32, default

Examples:
  whisper-ct2 -m ./whisper-base-ct2 audio.wav
  whisper-ct2 -m ./whisper-base-ct2 -l en -o srt audio.mp3
  whisper-ct2 -m ./whisper-base-ct2 --compute-type int8 long-audio.wav
```

### Output Formats

**Text** (default):
```
Hello world, this is a transcription test.
```

**JSON**:
```json
{
  "language": "en",
  "language_probability": 0.98,
  "duration": 5.5,
  "segments": [
    {
      "text": "Hello world, this is a transcription test.",
      "start": 0.0,
      "end": 5.5
    }
  ]
}
```

**SRT**:
```
1
00:00:00,000 --> 00:00:05,500
Hello world, this is a transcription test.
```

**WebVTT**:
```
WEBVTT

00:00:00.000 --> 00:00:05.500
Hello world, this is a transcription test.
```

## Build System

### Dependencies

**System packages (Arch Linux)**:
```bash
sudo pacman -S cmake base-devel pkgconf libsndfile libsamplerate openblas
```

**System packages (Ubuntu/Debian)**:
```bash
sudo apt install cmake build-essential pkg-config \
    libsndfile1-dev libsamplerate0-dev libopenblas-dev
```

**CTranslate2** (build from source):
```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2 && mkdir build && cd build
cmake .. -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Makefile Targets

```makefile
make build-cpp     # Build C++ library
make build-go      # Build Go package
make build-cli     # Build CLI binary (default)
make test          # Run all tests
make install-cpp   # Install C++ library system-wide
make clean         # Clean build artifacts
```

### Build Workflow

```
1. mkdir csrc/build && cd csrc/build
2. cmake .. -DCMAKE_BUILD_TYPE=Release
3. make -j$(nproc)
4. cd ../..
5. CGO_LDFLAGS="-L$(pwd)/csrc/build" go build ./cmd/whisper-ct2
```

## Model Conversion

Whisper models must be converted to CTranslate2 format:

```bash
pip install ctranslate2 transformers[torch]

# Convert with float32 (highest quality)
ct2-transformers-converter --model openai/whisper-base \
    --output_dir whisper-base-ct2 \
    --quantization float32

# Convert with int8 quantization (faster, smaller)
ct2-transformers-converter --model openai/whisper-base \
    --output_dir whisper-base-ct2-int8 \
    --quantization int8

# Available models:
# openai/whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3
```

## Performance Considerations

### Threading

- **inter_threads**: Batch parallelization (default: 1)
- **intra_threads**: Operations within a batch (default: 0 = auto, uses all cores)

For single-file transcription, `intra_threads=0` is optimal.
For batch processing, increase `inter_threads`.

### Quantization

| Compute Type | Size | Speed | Quality |
|--------------|------|-------|---------|
| float32 | 100% | Baseline | Best |
| float16 | 50% | ~1.5x | Excellent |
| int8 | 25% | ~2x | Good |

### Memory Usage

Model memory is approximately:
- tiny: ~150 MB
- base: ~300 MB
- small: ~1 GB
- medium: ~3 GB
- large-v3: ~6 GB

## Testing Strategy

### C++ Tests

```cpp
// Audio processor tests
TEST(AudioProcessorTest, LoadWavFile)
TEST(AudioProcessorTest, ResampleTo16kHz)
TEST(AudioProcessorTest, MelSpectrogramShape)
TEST(AudioProcessorTest, MelSpectrogramValues)  // Compare to faster-whisper

// STFT tests
TEST(STFTTest, HannWindow)
TEST(STFTTest, FFTOutput)
```

### Go Tests

```go
func TestLoadModel(t *testing.T)
func TestTranscribeFile(t *testing.T)
func TestTranscribePCM(t *testing.T)
func TestDetectLanguage(t *testing.T)
func TestWithOptions(t *testing.T)
func TestErrorHandling(t *testing.T)
```

### Integration Tests

- Compare transcription output to faster-whisper reference
- Test with various audio formats (WAV, MP3, FLAC)
- Test with different languages
- Benchmark against faster-whisper

## Future Enhancements

1. **GPU Support**: Add CUDA backend support
2. **Streaming**: Real-time audio transcription
3. **Word Timestamps**: Per-word timing information
4. **VAD Integration**: Voice activity detection for better segmentation
5. **Batch Processing**: Multiple files in parallel
