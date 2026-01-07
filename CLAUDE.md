# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make                # Build everything (C++ lib, Go package, CLI)
make build-cpp      # Build C++ library only
make build-go       # Build Go package only
make build-cli      # Build CLI binary
make test           # Run Go tests
make test-cpp       # Run C++ tests
make clean          # Remove build artifacts
make fmt            # Format Go code
make lint           # Run golangci-lint
sudo make install-cpp  # Install C++ library system-wide
```

The build requires CTranslate2 to be installed system-wide. See README.md for installation instructions.

## Architecture

This is a Go binding to CTranslate2 for Whisper speech-to-text inference without Python.

**Three-layer architecture:**
1. **C++ library** (`csrc/`) - Audio processing and CTranslate2 wrapper
   - `libwhisper_ct2.so` exposes a C API for Go cgo bindings
   - Audio pipeline: libsndfile (load) -> libsamplerate (resample to 16kHz) -> PocketFFT (STFT) -> mel spectrogram
   - Mel computation matches faster-whisper exactly (n_fft=400, hop=160, 80/128 mel bins)

2. **Go package** (`pkg/whisper/`) - High-level Go API
   - Uses cgo to call the C library
   - Functional options pattern for transcription configuration
   - Thread-safe: Model instances can be used concurrently

3. **CLI** (`cmd/whisper-ct2/`) - Command-line tool

**Key files:**
- `csrc/include/whisper_ct2.h` - Public C API header
- `csrc/src/whisper_ct2.cpp` - C API implementation calling CTranslate2
- `csrc/src/audio_processor.cpp` - Audio loading and mel spectrogram generation
- `pkg/whisper/whisper.go` - cgo bindings and version functions
- `pkg/whisper/model.go` - Model loading and lifecycle
- `pkg/whisper/transcribe.go` - Transcription and language detection

## cgo Considerations

The Go package links against the C++ library via cgo. Build flags are set in `pkg/whisper/whisper.go`:
```go
#cgo CFLAGS: -I${SRCDIR}/../../csrc/include
#cgo LDFLAGS: -L${SRCDIR}/../../csrc/build -lwhisper_ct2 -lstdc++ -lm
```

When running without system-wide installation, set `LD_LIBRARY_PATH` to include `csrc/build`.

## Error Handling

The C layer uses dual error reporting:
- Return codes (`whisper_error_t` enum)
- Thread-local detailed messages via `whisper_get_last_error_message()`

Go wraps these into typed sentinel errors (e.g., `ErrModelNotFound`, `ErrAudioLoadFailed`) in `pkg/whisper/errors.go`.

## Testing

Tests require a CTranslate2 Whisper model. The test expects `testdata/jfk.wav` as a test fixture.

Run a single Go test:
```bash
CGO_CFLAGS="-I$(pwd)/csrc/include" \
CGO_LDFLAGS="-L$(pwd)/csrc/build -lwhisper_ct2" \
LD_LIBRARY_PATH=$(pwd)/csrc/build \
go test -v -run TestTranscribeFile ./pkg/whisper/
```
