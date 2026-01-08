# whisper-benchmark

A performance benchmarking tool for `go-whisper-ct2`, similar to faster-whisper's `speed_benchmark.py`.

## Features

- **Multiple iterations**: Run transcription multiple times and get statistical analysis
- **Multiple audio files**: Benchmark several files in one run
- **Comprehensive statistics**: Min, max, mean, median, and standard deviation
- **Real-Time Factor (RTF)**: Automatic calculation of transcription speed relative to audio duration
- **Export results**: Save benchmark data to JSON or CSV for analysis
- **Warmup runs**: Optional warmup iteration to stabilize performance
- **Detailed logging**: Verbose mode for iteration-by-iteration results

## Installation

```bash
# Build the benchmark tool
make build-benchmark

# The binary will be in bin/whisper-benchmark
./bin/whisper-benchmark --help
```

## Usage

### Basic Benchmark

```bash
# Benchmark a single audio file (3 iterations by default)
./bin/whisper-benchmark --model ./whisper-base-ct2 audio.wav
```

### Multiple Iterations

```bash
# Run 10 iterations for more accurate statistics
./bin/whisper-benchmark --model ./whisper-base-ct2 --iterations 10 audio.wav
```

### Multiple Files

```bash
# Benchmark several audio files
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

### Verbose Mode

```bash
# Show detailed output for each iteration
./bin/whisper-benchmark \
  --model ./whisper-base-ct2 \
  --verbose \
  --iterations 5 \
  audio.wav
```

### Custom Configuration

```bash
# Use specific settings
./bin/whisper-benchmark \
  --model ./whisper-base-ct2 \
  --beam-size 10 \
  --threads 8 \
  --language en \
  --iterations 5 \
  audio.wav
```

## Output Metrics

The benchmark reports several key metrics:

### Transcription Time Statistics
- **Min**: Fastest transcription time
- **Max**: Slowest transcription time
- **Mean**: Average transcription time across all iterations
- **Median**: Middle value when times are sorted
- **Std Dev**: Standard deviation showing consistency

### Real-Time Factor (RTF)
The ratio of audio duration to transcription time. **Higher is better.**

- **RTF = audio_duration / transcription_time**
- Example: RTF of 3.5x means the system processes 3.5 seconds of audio per second of computation
- RTF > 1.0 = faster than real-time
- RTF < 1.0 = slower than real-time

### Other Metrics
- Language detection results
- Number of segments
- Text length (characters)

## Example Output

```
Loading model from ./whisper-base-ct2...
Model loaded in 1.23s
  Multilingual: true
  Mel bins: 80

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

================================================================================
SUMMARY
================================================================================
Total audio duration:       4.82s
Total transcription time:   1.27s (mean)
Overall RTF:                3.81x
Files processed:            1
Iterations per file:        3
================================================================================
```

## Comparison with faster-whisper

| Feature | whisper-benchmark | faster-whisper benchmark |
|---------|-------------------|--------------------------|
| Multiple iterations | ✅ Yes | ✅ Yes (via timeit) |
| Statistical analysis | ✅ Min, max, mean, median, stddev | ⚠️ Min only |
| RTF calculation | ✅ Automatic | ❌ Manual |
| Multiple files | ✅ Yes | ❌ One at a time |
| JSON export | ✅ Yes | ❌ No |
| CSV export | ✅ Yes | ❌ No |
| Warmup runs | ✅ Yes (optional) | ❌ No |
| Verbose mode | ✅ Yes | ❌ No |

## JSON Output Format

```json
{
  "model_path": "./whisper-base-ct2",
  "compute_type": "default",
  "beam_size": 5,
  "threads": 0,
  "language": "auto",
  "task": "transcribe",
  "timestamp": "2025-01-08T10:30:00Z",
  "file_stats": [
    {
      "audio_file": "audio.wav",
      "iterations": 3,
      "audio_duration_sec": 4.82,
      "min_time_sec": 1.234,
      "max_time_sec": 1.298,
      "mean_time_sec": 1.267,
      "median_time_sec": 1.271,
      "std_dev_sec": 0.027,
      "min_rtf": 3.71,
      "max_rtf": 3.91,
      "mean_rtf": 3.81,
      "median_rtf": 3.79,
      "all_results": [...]
    }
  ]
}
```

## CSV Output Format

The CSV file contains one row per audio file with all statistics:

```csv
audio_file,iterations,audio_duration_sec,min_time_sec,max_time_sec,mean_time_sec,median_time_sec,std_dev_sec,min_rtf,max_rtf,mean_rtf,median_rtf,language,segments,text_length
audio.wav,3,4.82,1.234,1.298,1.267,1.271,0.027,3.71,3.91,3.81,3.79,en,2,142
```

## Tips for Accurate Benchmarks

1. **Use warmup**: The default warmup run helps stabilize performance (enabled by default)
2. **Multiple iterations**: Use at least 3-5 iterations for reliable statistics
3. **Close other programs**: Minimize background processes for consistent results
4. **Use representative audio**: Test with audio files similar to your use case
5. **Check CPU throttling**: Ensure your CPU isn't thermally throttling
6. **Compare configurations**: Test different beam sizes, thread counts, and compute types

## Benchmarking Different Configurations

```bash
# Compare int8 vs default quantization
./bin/whisper-benchmark --model ./whisper-base-ct2-int8 --compute-type int8 --output-json int8.json audio.wav
./bin/whisper-benchmark --model ./whisper-base-ct2 --output-json default.json audio.wav

# Compare different beam sizes
for beam in 1 5 10; do
  ./bin/whisper-benchmark --model ./whisper-base-ct2 --beam-size $beam --output-json beam-$beam.json audio.wav
done

# Compare thread counts
for threads in 1 4 8 16; do
  ./bin/whisper-benchmark --model ./whisper-base-ct2 --threads $threads --output-json threads-$threads.json audio.wav
done
```

## License

MIT License (same as go-whisper-ct2)
