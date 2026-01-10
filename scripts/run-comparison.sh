#!/bin/bash
#
# Run full comparison between go-whisper-ct2 and faster-whisper
#
# Usage:
#   ./run-comparison.sh --model ./whisper-base-ct2 audio.wav
#

set -e

# Default values
MODEL=""
ITERATIONS=5
AUDIO_FILES=()
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --model <model-path> [options] <audio-files...>"
            echo ""
            echo "Options:"
            echo "  --model PATH        Path to CTranslate2 model (required)"
            echo "  --iterations N      Number of iterations (default: 5)"
            echo "  --verbose           Show detailed output"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --model ./whisper-base-ct2 --iterations 10 audio.wav"
            exit 0
            ;;
        *)
            AUDIO_FILES+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [ ${#AUDIO_FILES[@]} -eq 0 ]; then
    echo "Error: at least one audio file is required"
    echo "Run with --help for usage information"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL" ]; then
    echo "Error: Model directory not found: $MODEL"
    exit 1
fi

# Check if audio files exist
for file in "${AUDIO_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Audio file not found: $file"
        exit 1
    fi
done

# Check if Go benchmark exists
if [ ! -f "./bin/whisper-benchmark" ]; then
    echo "Error: Go benchmark not built"
    echo "Run: make build-benchmark"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if faster-whisper is installed
if ! python3 -c "import faster_whisper" 2>/dev/null; then
    echo "Error: faster-whisper not installed"
    echo "Install with: pip install faster-whisper"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================"
echo "Go vs Python Benchmark Comparison"
echo "========================================"
echo ""
echo "Model:      $MODEL"
echo "Iterations: $ITERATIONS"
echo "Files:      ${AUDIO_FILES[@]}"
echo ""

# 1. Run Go benchmark
echo "========================================"
echo "Running go-whisper-ct2 benchmark..."
echo "========================================"
echo ""

./bin/whisper-benchmark \
    --model "$MODEL" \
    --iterations "$ITERATIONS" \
    --output-json go-benchmark.json \
    $VERBOSE \
    "${AUDIO_FILES[@]}"

echo ""
echo "✅ Go benchmark complete!"
echo ""

# 2. Run Python benchmark
echo "========================================"
echo "Running faster-whisper benchmark..."
echo "========================================"
echo ""

python3 "$SCRIPT_DIR/benchmark-python.py" \
    --model "$MODEL" \
    --iterations "$ITERATIONS" \
    --output-json python-benchmark.json \
    $VERBOSE \
    "${AUDIO_FILES[@]}"

echo ""
echo "✅ Python benchmark complete!"
echo ""

# 3. Compare results
echo "========================================"
echo "Comparing Results..."
echo "========================================"
echo ""

python3 "$SCRIPT_DIR/compare.py" go-benchmark.json python-benchmark.json

echo ""
echo "========================================"
echo "Benchmark files saved:"
echo "========================================"
echo "  - go-benchmark.json"
echo "  - python-benchmark.json"
echo ""
