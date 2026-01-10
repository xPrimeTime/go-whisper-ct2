#!/usr/bin/env python3
"""
Benchmark faster-whisper Python implementation.
Matches the format of go-whisper-ct2 benchmark tool.

Usage:
    python3 benchmark-python.py --model ./whisper-base-ct2 --iterations 5 audio.wav
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed")
    print("Install with: pip install faster-whisper")
    sys.exit(1)


def benchmark_file(model, audio_file, iterations, warmup, verbose):
    """Benchmark a single audio file."""
    print(f"Benchmarking: {Path(audio_file).name}")
    print("=" * 80)

    results = []

    # Warmup run
    if warmup:
        if verbose:
            print("Running warmup iteration...")
        try:
            segments, info = model.transcribe(audio_file, beam_size=5, language=None)
            list(segments)  # Force evaluation
        except Exception as e:
            print(f"Warning: warmup failed: {e}")

    # Run benchmark iterations
    audio_duration = None

    for i in range(iterations):
        if verbose:
            print(f"Iteration {i+1}/{iterations}...")

        start = time.time()
        segments, info = model.transcribe(audio_file, beam_size=5, language=None)
        segments_list = list(segments)  # Force evaluation
        elapsed = time.time() - start

        # Get audio duration from info
        if audio_duration is None:
            audio_duration = info.duration

        result = {
            "iteration": i + 1,
            "time": elapsed,
            "audio_duration": audio_duration,
            "language": info.language,
            "language_probability": info.language_probability,
        }
        results.append(result)

        if verbose:
            rtf = audio_duration / elapsed if elapsed > 0 else 0
            print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.2f}x | Lang: {info.language}")

    if not results:
        print(f"Error: no successful iterations for {audio_file}")
        return None

    # Calculate statistics
    times = [r["time"] for r in results]
    rtfs = [r["audio_duration"] / r["time"] if r["time"] > 0 else 0 for r in results]

    stats = {
        "audio_file": str(audio_file),
        "iterations": len(results),
        "audio_duration_sec": results[0]["audio_duration"],
        "min_time": min(times),
        "max_time": max(times),
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_rtf": max(rtfs),  # Min time = max RTF
        "max_rtf": min(rtfs),  # Max time = min RTF
        "mean_rtf": statistics.mean(rtfs),
        "median_rtf": statistics.median(rtfs),
        "language": results[0]["language"],
        "language_probability": results[0]["language_probability"],
    }

    return stats


def print_stats(stats):
    """Print statistics for a file."""
    if stats is None:
        return

    print()
    print(f"Audio: {Path(stats['audio_file']).name}")
    print(f"  Duration:           {stats['audio_duration_sec']:.2f}s")
    print(f"  Iterations:         {stats['iterations']}")
    print()
    print("Transcription Time:")
    print(f"  Min:                {stats['min_time']:.3f}s")
    print(f"  Max:                {stats['max_time']:.3f}s")
    print(f"  Mean:               {stats['mean_time']:.3f}s")
    print(f"  Median:             {stats['median_time']:.3f}s")
    print(f"  Std Dev:            {stats['std_dev']:.3f}s")
    print()
    print("Real-Time Factor:")
    print(f"  Min RTF:            {stats['min_rtf']:.2f}x (fastest)")
    print(f"  Max RTF:            {stats['max_rtf']:.2f}x (slowest)")
    print(f"  Mean RTF:           {stats['mean_rtf']:.2f}x")
    print(f"  Median RTF:         {stats['median_rtf']:.2f}x")
    print()
    print("Transcription Info:")
    print(f"  Language:           {stats['language']}")
    print(f"  Language prob:      {stats['language_probability']:.2%}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark faster-whisper performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("audio_files", nargs="+", help="Audio files to benchmark")
    parser.add_argument("--model", required=True, help="Path to CTranslate2 model directory")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations (default: 3)")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run warmup iteration (default: true)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup", help="Disable warmup iteration")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--output-json", help="Save results to JSON file")
    parser.add_argument("--compute-type", default="default", help="Compute type (default, int8, float16, float32)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    # Check audio files exist
    for audio_file in args.audio_files:
        if not Path(audio_file).exists():
            print(f"Error: Audio file not found: {audio_file}")
            sys.exit(1)

    # Load model
    print(f"Loading model from {args.model}...")
    start = time.time()
    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Benchmark each file
    all_stats = []
    for audio_file in args.audio_files:
        stats = benchmark_file(model, audio_file, args.iterations, args.warmup, args.verbose)
        if stats:
            all_stats.append(stats)
            print_stats(stats)

    # Save results
    if args.output_json and all_stats:
        report = {
            "model_path": args.model,
            "compute_type": args.compute_type,
            "device": args.device,
            "iterations": args.iterations,
            "file_stats": all_stats,
        }

        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # Print summary
    if len(all_stats) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_duration = sum(s["audio_duration_sec"] for s in all_stats)
        total_time = sum(s["mean_time"] for s in all_stats)
        overall_rtf = total_duration / total_time if total_time > 0 else 0

        print(f"Total audio duration:       {total_duration:.2f}s")
        print(f"Total transcription time:   {total_time:.2f}s (mean)")
        print(f"Overall RTF:                {overall_rtf:.2f}x")
        print(f"Files processed:            {len(all_stats)}")
        print(f"Iterations per file:        {args.iterations}")
        print("=" * 80)


if __name__ == "__main__":
    main()
