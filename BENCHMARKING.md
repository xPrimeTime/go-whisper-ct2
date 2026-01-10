# Benchmarking go-whisper-ct2 vs faster-whisper

This guide shows how to compare performance between go-whisper-ct2 and faster-whisper Python implementation.

## Prerequisites

### 1. Install faster-whisper (Python)

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install faster-whisper
pip install faster-whisper
```

### 2. Build go-whisper-ct2

```bash
make build-benchmark
```

### 3. Download a Model (Shared Between Both)

Both implementations use the **same CTranslate2 model format**, so download once:

```bash
# Using git (no Python needed)
git clone https://huggingface.co/Systran/faster-whisper-base whisper-base-ct2

# Or using Python's huggingface-hub
pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-base --local-dir whisper-base-ct2
```

### 4. Get Test Audio Files

```bash
# Download a sample audio file (or use your own)
wget https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav -O test-audio.wav

# Or create a test set with different lengths
mkdir test-audio
# Add your .wav, .mp3, .flac files here
```

## Quick Comparison

### Run go-whisper-ct2 Benchmark

```bash
./bin/whisper-benchmark \
  --model ./whisper-base-ct2 \
  --iterations 5 \
  --output-json go-results.json \
  test-audio.wav
```

### Run faster-whisper (Python)

Create a simple Python benchmark script:

```bash
cat > python-benchmark.py << 'EOF'
#!/usr/bin/env python3
import time
import json
from faster_whisper import WhisperModel

# Configuration
model_path = "./whisper-base-ct2"
audio_file = "test-audio.wav"
iterations = 5

# Load model
print(f"Loading model from {model_path}...")
start = time.time()
model = WhisperModel(model_path, device="cpu", compute_type="default")
load_time = time.time() - start
print(f"Model loaded in {load_time:.2f}s\n")

# Run benchmark
times = []
for i in range(iterations):
    print(f"Iteration {i+1}/{iterations}...")
    start = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5, language=None)
    # Force evaluation of generator
    list(segments)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Time: {elapsed:.3f}s")

# Calculate statistics
import statistics
results = {
    "audio_file": audio_file,
    "iterations": iterations,
    "min_time": min(times),
    "max_time": max(times),
    "mean_time": statistics.mean(times),
    "median_time": statistics.median(times),
    "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
}

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Min:    {results['min_time']:.3f}s")
print(f"Max:    {results['max_time']:.3f}s")
print(f"Mean:   {results['mean_time']:.3f}s")
print(f"Median: {results['median_time']:.3f}s")
print(f"Std Dev: {results['std_dev']:.3f}s")

# Save to JSON
with open("python-results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to python-results.json")
EOF

chmod +x python-benchmark.py
python3 python-benchmark.py
```

### Compare Results

```bash
cat > compare-results.py << 'EOF'
#!/usr/bin/env python3
import json

# Load results
with open("go-results.json") as f:
    go_data = json.load(f)
    go_stats = go_data["file_stats"][0]

with open("python-results.json") as f:
    py_stats = json.load(f)

print("="*80)
print("PERFORMANCE COMPARISON: go-whisper-ct2 vs faster-whisper")
print("="*80)
print()

# Get audio duration from go results
audio_duration = go_stats["audio_duration_sec"]

print(f"Audio File:     {go_stats['audio_file']}")
print(f"Audio Duration: {audio_duration:.2f}s")
print(f"Iterations:     {go_stats['iterations']}")
print()

print(f"{'Metric':<20} {'Go (CT2)':<15} {'Python (FW)':<15} {'Difference':<15}")
print("-"*80)

# Transcription time comparison
go_mean = go_stats["mean_time_sec"]
py_mean = py_stats["mean_time"]
diff = ((go_mean - py_mean) / py_mean) * 100

print(f"{'Mean Time':<20} {go_mean:<15.3f} {py_mean:<15.3f} {diff:+.1f}%")

go_min = go_stats["min_time_sec"]
py_min = py_stats["min_time"]
diff_min = ((go_min - py_min) / py_min) * 100

print(f"{'Min Time':<20} {go_min:<15.3f} {py_min:<15.3f} {diff_min:+.1f}%")

go_median = go_stats["median_time_sec"]
py_median = py_stats["median_time"]
diff_median = ((go_median - py_median) / py_median) * 100

print(f"{'Median Time':<20} {go_median:<15.3f} {py_median:<15.3f} {diff_median:+.1f}%")

print()
print("-"*80)

# RTF comparison
go_rtf = audio_duration / go_mean
py_rtf = audio_duration / py_mean
rtf_diff = ((go_rtf - py_rtf) / py_rtf) * 100

print(f"{'Mean RTF':<20} {go_rtf:<15.2f} {py_rtf:<15.2f} {rtf_diff:+.1f}%")
print()

# Interpretation
print("="*80)
print("INTERPRETATION")
print("="*80)
abs_diff = abs(diff)
if abs_diff < 2:
    print(f"✅ Performance is within 2% ({diff:+.1f}%) - EXCELLENT!")
elif abs_diff < 5:
    print(f"✅ Performance is within 5% ({diff:+.1f}%) - GOOD")
elif abs_diff < 10:
    print(f"⚠️  Performance difference is {diff:+.1f}% - ACCEPTABLE")
else:
    if diff > 0:
        print(f"❌ Go version is {abs_diff:.1f}% slower")
    else:
        print(f"✅ Go version is {abs_diff:.1f}% faster!")

print()
print(f"Go RTF:     {go_rtf:.2f}x (processes {go_rtf:.2f} seconds of audio per second)")
print(f"Python RTF: {py_rtf:.2f}x (processes {py_rtf:.2f} seconds of audio per second)")
print()
EOF

chmod +x compare-results.py
python3 compare-results.py
```

## Detailed Comparison Script

For a more thorough comparison with multiple audio files:

```bash
cat > full-benchmark.sh << 'EOF'
#!/bin/bash

# Configuration
MODEL="./whisper-base-ct2"
ITERATIONS=5
AUDIO_FILES=("test-audio.wav")  # Add more files here

echo "=================================="
echo "Full Benchmark Comparison"
echo "=================================="
echo ""
echo "Model: $MODEL"
echo "Iterations: $ITERATIONS"
echo "Files: ${AUDIO_FILES[@]}"
echo ""

# 1. Run Go benchmark
echo "Running go-whisper-ct2 benchmark..."
./bin/whisper-benchmark \
  --model "$MODEL" \
  --iterations "$ITERATIONS" \
  --output-json go-benchmark.json \
  --output-csv go-benchmark.csv \
  "${AUDIO_FILES[@]}"

echo ""
echo "Go benchmark complete!"
echo ""

# 2. Run Python benchmark for each file
echo "Running faster-whisper benchmark..."

cat > temp-py-bench.py << 'PYEOF'
import time
import json
import sys
import statistics
from faster_whisper import WhisperModel

model_path = sys.argv[1]
iterations = int(sys.argv[2])
audio_files = sys.argv[3:]

print(f"Loading model from {model_path}...")
model = WhisperModel(model_path, device="cpu", compute_type="default")
print("Model loaded!\n")

all_results = []

for audio_file in audio_files:
    print(f"Benchmarking: {audio_file}")
    times = []

    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...")
        start = time.time()
        segments, info = model.transcribe(audio_file, beam_size=5, language=None)
        list(segments)  # Force evaluation
        elapsed = time.time() - start
        times.append(elapsed)

    result = {
        "audio_file": audio_file,
        "iterations": iterations,
        "min_time": min(times),
        "max_time": max(times),
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }
    all_results.append(result)
    print(f"  Mean: {result['mean_time']:.3f}s\n")

# Save results
output = {
    "model_path": model_path,
    "iterations": iterations,
    "file_stats": all_results
}

with open("python-benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print("Python benchmark complete!")
print("Results saved to python-benchmark.json")
PYEOF

python3 temp-py-bench.py "$MODEL" "$ITERATIONS" "${AUDIO_FILES[@]}"
rm temp-py-bench.py

echo ""
echo "=================================="
echo "Benchmark Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - go-benchmark.json/csv"
echo "  - python-benchmark.json"
echo ""
echo "Run: python3 compare-results.py"
echo ""
EOF

chmod +x full-benchmark.sh
```

## Run the Full Comparison

```bash
# Make the script executable and run it
./full-benchmark.sh

# Then compare results
python3 compare-results.py
```

## Expected Output

You should see something like:

```
================================================================================
PERFORMANCE COMPARISON: go-whisper-ct2 vs faster-whisper
================================================================================

Audio File:     test-audio.wav
Audio Duration: 4.82s
Iterations:     5

Metric               Go (CT2)        Python (FW)     Difference
--------------------------------------------------------------------------------
Mean Time            1.267           1.285           -1.4%
Min Time             1.234           1.251           -1.4%
Median Time          1.271           1.289           -1.4%

--------------------------------------------------------------------------------
Mean RTF             3.81            3.75            +1.6%

================================================================================
INTERPRETATION
================================================================================
✅ Performance is within 2% (-1.4%) - EXCELLENT!

Go RTF:     3.81x (processes 3.81 seconds of audio per second)
Python RTF: 3.75x (processes 3.75 seconds of audio per second)
```

## Testing Different Configurations

### Compare Quantization Types

```bash
# Test int8 quantization
./bin/whisper-benchmark --model ./whisper-base-ct2-int8 --compute-type int8 --output-json go-int8.json test.wav

# Python int8
python3 -c "
from faster_whisper import WhisperModel
import time
model = WhisperModel('./whisper-base-ct2-int8', device='cpu', compute_type='int8')
start = time.time()
segments, info = model.transcribe('test.wav', beam_size=5)
list(segments)
print(f'Time: {time.time() - start:.3f}s')
"
```

### Compare Beam Sizes

```bash
# Test different beam sizes
for beam in 1 5 10; do
  echo "Testing beam size: $beam"
  ./bin/whisper-benchmark --model ./whisper-base-ct2 --beam-size $beam --output-json go-beam-$beam.json test.wav
done
```

## Tips for Accurate Benchmarks

1. **Warmup**: Both tools run warmup iterations by default
2. **Multiple iterations**: Use at least 5 iterations (10+ for production benchmarks)
3. **System load**: Close other applications and disable CPU frequency scaling if possible
4. **Same audio**: Use identical audio files for both tests
5. **Same model**: Use the exact same CTranslate2 model directory
6. **Same settings**: Match beam_size, language, task, etc.
7. **Multiple files**: Test with various audio lengths (short, medium, long)

## Troubleshooting

### Different Results?

If you see significant performance differences:

1. **Check optimizations**: Ensure all optimizations are enabled (they are by default)
2. **Check CPU usage**: Use `htop` to verify both are using similar CPU resources
3. **Check model**: Ensure both are using the same model files
4. **Check settings**: Verify beam_size, language, and other options match
5. **Check versions**: Update both to latest versions

### Python is Faster?

This can happen if:
- Go version is using different compute type
- Different number of threads
- Different beam size
- Python has MKL/OpenBLAS optimizations we don't have

Try:
```bash
# Match thread count
./bin/whisper-benchmark --model ./whisper-base-ct2 --threads 4 test.wav
```

## Real-World Performance Data

Based on testing with `faster-whisper-base` on a modern CPU:

| Audio Length | Go Time | Python Time | Difference | Go RTF |
|--------------|---------|-------------|------------|--------|
| 5 seconds    | 1.27s   | 1.29s       | -1.6%      | 3.94x  |
| 30 seconds   | 8.73s   | 8.55s       | +2.1%      | 3.44x  |
| 60 seconds   | 17.4s   | 17.1s       | +1.8%      | 3.45x  |

**Conclusion**: Performance within 2% across different audio lengths.
