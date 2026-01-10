# Benchmark Comparison Scripts

This directory contains scripts for comparing go-whisper-ct2 performance against faster-whisper (Python).

## Quick Start

### 1. Install faster-whisper

```bash
pip install faster-whisper
```

### 2. Run Comparison

```bash
# One-command comparison (recommended)
./scripts/run-comparison.sh --model ./whisper-base-ct2 audio.wav
```

This will:
1. Run go-whisper-ct2 benchmark (5 iterations by default)
2. Run faster-whisper Python benchmark (same settings)
3. Compare and display results side-by-side

## Available Scripts

### `run-comparison.sh` - Complete Automated Comparison

Runs both benchmarks and compares results automatically.

```bash
# Basic usage
./scripts/run-comparison.sh --model ./whisper-base-ct2 audio.wav

# Multiple files
./scripts/run-comparison.sh --model ./whisper-base-ct2 audio1.wav audio2.wav

# More iterations for accuracy
./scripts/run-comparison.sh --model ./whisper-base-ct2 --iterations 10 audio.wav

# Verbose output
./scripts/run-comparison.sh --model ./whisper-base-ct2 --verbose audio.wav
```

**Output files:**
- `go-benchmark.json` - Go benchmark results
- `python-benchmark.json` - Python benchmark results

### `benchmark-python.py` - Python Benchmark Tool

Benchmarks faster-whisper with the same interface as the Go tool.

```bash
# Basic usage
./scripts/benchmark-python.py --model ./whisper-base-ct2 --iterations 5 audio.wav

# Save results to JSON
./scripts/benchmark-python.py \
    --model ./whisper-base-ct2 \
    --iterations 5 \
    --output-json results.json \
    audio.wav

# Multiple files
./scripts/benchmark-python.py \
    --model ./whisper-base-ct2 \
    --iterations 5 \
    audio1.wav audio2.wav audio3.wav

# Verbose mode
./scripts/benchmark-python.py \
    --model ./whisper-base-ct2 \
    --verbose \
    --iterations 5 \
    audio.wav

# With int8 quantization
./scripts/benchmark-python.py \
    --model ./whisper-base-ct2-int8 \
    --compute-type int8 \
    audio.wav
```

**Options:**
- `--model PATH` - Path to CTranslate2 model directory (required)
- `--iterations N` - Number of iterations (default: 3)
- `--output-json FILE` - Save results to JSON
- `--warmup` / `--no-warmup` - Enable/disable warmup run (default: enabled)
- `--verbose` - Show detailed output per iteration
- `--compute-type TYPE` - Compute type: default, int8, float16, float32
- `--device TYPE` - Device: cpu or cuda

### `compare.py` - Results Comparison Tool

Compares benchmark JSON results and displays statistics.

```bash
# Compare two benchmark results
./scripts/compare.py go-benchmark.json python-benchmark.json
```

**Input:** Two JSON files from benchmarks (Go and Python)

**Output:**
- Side-by-side comparison of metrics
- Performance difference percentages
- RTF (Real-Time Factor) analysis
- Interpretation (within 2%, 5%, 10%, etc.)

## Example Workflow

### Full Comparison

```bash
# 1. Build Go benchmark
make build-benchmark

# 2. Download a model (if you don't have one)
git clone https://huggingface.co/Systran/faster-whisper-base whisper-base-ct2

# 3. Get test audio
wget https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav

# 4. Run comparison
./scripts/run-comparison.sh --model ./whisper-base-ct2 --iterations 10 jfk.wav
```

### Manual Step-by-Step

```bash
# 1. Run Go benchmark
./bin/whisper-benchmark \
    --model ./whisper-base-ct2 \
    --iterations 5 \
    --output-json go-results.json \
    audio.wav

# 2. Run Python benchmark
./scripts/benchmark-python.py \
    --model ./whisper-base-ct2 \
    --iterations 5 \
    --output-json python-results.json \
    audio.wav

# 3. Compare results
./scripts/compare.py go-results.json python-results.json
```

## Example Output

```
================================================================================
PERFORMANCE COMPARISON: go-whisper-ct2 vs faster-whisper
================================================================================

Audio File:     jfk.wav
Audio Duration: 11.00s
Iterations:     5

Metric               Go              Python          Difference
--------------------------------------------------------------------------------
Mean Time            3.189           3.215           -0.8%
Min Time             3.156           3.198           -1.3%
Max Time             3.234           3.245           -0.3%
Median Time          3.187           3.212           -0.8%
Std Dev              0.028           0.019           +47.4%

--------------------------------------------------------------------------------
Mean RTF             3.45            3.42            +0.9%

================================================================================
INTERPRETATION
================================================================================
✅ Performance is within 2% (-0.8%) - EXCELLENT!
   Go and Python implementations are virtually identical in speed.

Go RTF:     3.45x (processes 3.45 seconds of audio per second)
Python RTF: 3.42x (processes 3.42 seconds of audio per second)

✅ Go: Faster than real-time (3.5x)
✅ Python: Faster than real-time (3.4x)
```

## Testing Different Configurations

### Compare Quantization Types

```bash
# Test int8 vs default
./scripts/run-comparison.sh --model ./whisper-base-ct2-int8 audio.wav > int8-results.txt
./scripts/run-comparison.sh --model ./whisper-base-ct2 audio.wav > default-results.txt
```

### Compare Multiple Audio Lengths

```bash
# Create test set
mkdir test-audio
# Add short, medium, long audio files

# Run comprehensive test
for audio in test-audio/*.wav; do
    echo "Testing: $audio"
    ./scripts/run-comparison.sh --model ./whisper-base-ct2 "$audio"
    echo ""
done
```

### Compare Different Models

```bash
# Base model
./scripts/run-comparison.sh --model ./whisper-base-ct2 audio.wav > base.txt

# Small model
./scripts/run-comparison.sh --model ./whisper-small-ct2 audio.wav > small.txt

# Medium model
./scripts/run-comparison.sh --model ./whisper-medium-ct2 audio.wav > medium.txt
```

## Troubleshooting

### "faster-whisper not installed"

```bash
pip install faster-whisper
```

### "whisper-benchmark not found"

```bash
make build-benchmark
```

### Different Performance Results

If Go and Python show significantly different speeds:

1. **Check they're using the same model**
   ```bash
   ls -la whisper-base-ct2/
   # Both should point to same directory
   ```

2. **Check CPU usage during benchmark**
   ```bash
   htop  # Run in another terminal during benchmark
   ```

3. **Ensure both use same settings**
   - Same beam size (default: 5)
   - Same language detection (default: auto)
   - Same number of threads

4. **Try more iterations**
   ```bash
   ./scripts/run-comparison.sh --model ./whisper-base-ct2 --iterations 10 audio.wav
   ```

### Python Benchmark Crashes

Check Python environment:
```bash
python3 --version  # Should be 3.8+
pip list | grep faster-whisper
```

## Performance Tips

1. **Use at least 5 iterations** for reliable statistics
2. **Close other applications** during benchmarking
3. **Test with multiple audio files** of different lengths
4. **Use warmup runs** (enabled by default)
5. **Compare same-length audio** for fairest comparison

## Files Generated

After running benchmarks:

- `go-benchmark.json` - Go benchmark results (detailed)
- `python-benchmark.json` - Python benchmark results (detailed)
- Individual result files if you use `--output-json`

You can keep these for future reference or analysis.

## Integration with CI/CD

Run automated comparisons in CI:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark comparison
  run: |
    ./scripts/run-comparison.sh \
      --model ./whisper-base-ct2 \
      test-audio.wav > benchmark-results.txt
```

## License

Same as go-whisper-ct2 (MIT License)
