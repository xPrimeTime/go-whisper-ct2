#!/usr/bin/env python3
"""
Compare benchmark results between go-whisper-ct2 and faster-whisper.

Usage:
    python3 compare.py go-results.json python-results.json
"""

import json
import sys
from pathlib import Path


def load_results(go_file, py_file):
    """Load results from JSON files."""
    with open(go_file) as f:
        go_data = json.load(f)
        go_stats = go_data["file_stats"][0]

    with open(py_file) as f:
        py_stats = json.load(f)

    return go_stats, py_stats


def print_comparison(go_stats, py_stats):
    """Print detailed comparison."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON: go-whisper-ct2 vs faster-whisper")
    print("=" * 80)
    print()

    # Get audio duration from go results
    audio_duration = go_stats["audio_duration_sec"]

    print(f"Audio File:     {Path(go_stats['audio_file']).name}")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Iterations:     {go_stats['iterations']}")
    print()

    print(f"{'Metric':<20} {'Go':<15} {'Python':<15} {'Difference':<15}")
    print("-" * 80)

    # Transcription time comparison
    metrics = [
        ("Mean Time", "mean_time_sec", "mean_time"),
        ("Min Time", "min_time_sec", "min_time"),
        ("Max Time", "max_time_sec", "max_time"),
        ("Median Time", "median_time_sec", "median_time"),
        ("Std Dev", "std_dev_sec", "std_dev"),
    ]

    for label, go_key, py_key in metrics:
        go_val = go_stats[go_key]
        py_val = py_stats[py_key]
        diff = ((go_val - py_val) / py_val) * 100 if py_val != 0 else 0
        print(f"{label:<20} {go_val:<15.3f} {py_val:<15.3f} {diff:+.1f}%")

    print()
    print("-" * 80)

    # RTF comparison
    go_mean = go_stats["mean_time_sec"]
    py_mean = py_stats["mean_time"]

    go_rtf = audio_duration / go_mean if go_mean != 0 else 0
    py_rtf = audio_duration / py_mean if py_mean != 0 else 0
    rtf_diff = ((go_rtf - py_rtf) / py_rtf) * 100 if py_rtf != 0 else 0

    print(f"{'Mean RTF':<20} {go_rtf:<15.2f} {py_rtf:<15.2f} {rtf_diff:+.1f}%")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    time_diff = ((go_mean - py_mean) / py_mean) * 100 if py_mean != 0 else 0
    abs_diff = abs(time_diff)

    if abs_diff < 2:
        print(f"✅ Performance is within 2% ({time_diff:+.1f}%) - EXCELLENT!")
        print("   Go and Python implementations are virtually identical in speed.")
    elif abs_diff < 5:
        print(f"✅ Performance is within 5% ({time_diff:+.1f}%) - GOOD")
        print("   Performance difference is acceptable for most use cases.")
    elif abs_diff < 10:
        print(f"⚠️  Performance difference is {time_diff:+.1f}% - ACCEPTABLE")
        print("   Noticeable difference but still reasonable.")
    else:
        if time_diff > 0:
            print(f"❌ Go version is {abs_diff:.1f}% slower")
            print("   Consider checking configuration and optimizations.")
        else:
            print(f"✅ Go version is {abs_diff:.1f}% faster!")
            print("   Excellent! Go implementation outperforms Python.")

    print()
    print(f"Go RTF:     {go_rtf:.2f}x (processes {go_rtf:.2f} seconds of audio per second)")
    print(f"Python RTF: {py_rtf:.2f}x (processes {py_rtf:.2f} seconds of audio per second)")
    print()

    if go_rtf > 1.0:
        print(f"✅ Go: Faster than real-time ({go_rtf:.1f}x)")
    else:
        print(f"❌ Go: Slower than real-time ({go_rtf:.1f}x)")

    if py_rtf > 1.0:
        print(f"✅ Python: Faster than real-time ({py_rtf:.1f}x)")
    else:
        print(f"❌ Python: Slower than real-time ({py_rtf:.1f}x)")

    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare.py <go-results.json> <python-results.json>")
        print()
        print("Example:")
        print("  python3 compare.py go-benchmark.json python-benchmark.json")
        sys.exit(1)

    go_file = sys.argv[1]
    py_file = sys.argv[2]

    if not Path(go_file).exists():
        print(f"Error: Go results file not found: {go_file}")
        sys.exit(1)

    if not Path(py_file).exists():
        print(f"Error: Python results file not found: {py_file}")
        sys.exit(1)

    try:
        go_stats, py_stats = load_results(go_file, py_file)
        print_comparison(go_stats, py_stats)
    except KeyError as e:
        print(f"Error: Invalid results file format - missing key: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
