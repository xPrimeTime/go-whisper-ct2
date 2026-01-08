package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/xPrimeTime/go-whisper-ct2/pkg/whisper"
)

var (
	modelPath   = flag.String("model", "", "Path to CTranslate2 model directory (required)")
	iterations  = flag.Int("iterations", 3, "Number of times to run each transcription")
	language    = flag.String("language", "auto", "Language code (e.g., en, fr) or 'auto' for detection")
	task        = flag.String("task", "transcribe", "Task: 'transcribe' or 'translate'")
	beamSize    = flag.Int("beam-size", 5, "Beam search width")
	computeType = flag.String("compute-type", "default", "Compute type: int8, int16, float16, float32, default")
	threads     = flag.Int("threads", 0, "Number of threads (0 = auto)")
	outputJSON  = flag.String("output-json", "", "Save results to JSON file")
	outputCSV   = flag.String("output-csv", "", "Save results to CSV file")
	warmup      = flag.Bool("warmup", true, "Run one warmup iteration (not counted in stats)")
	verbose     = flag.Bool("verbose", false, "Show detailed output for each iteration")
)

// BenchmarkResult holds the results of a single transcription run
type BenchmarkResult struct {
	AudioFile         string        `json:"audio_file"`
	AudioDuration     float64       `json:"audio_duration_sec"`
	TranscriptionTime time.Duration `json:"transcription_time"`
	TranscriptionSec  float64       `json:"transcription_sec"`
	RTF               float64       `json:"real_time_factor"`
	Language          string        `json:"language"`
	SegmentCount      int           `json:"segment_count"`
	TextLength        int           `json:"text_length"`
}

// BenchmarkStats holds statistical analysis of benchmark results
type BenchmarkStats struct {
	AudioFile     string          `json:"audio_file"`
	Iterations    int             `json:"iterations"`
	AudioDuration float64         `json:"audio_duration_sec"`
	MinTime       float64         `json:"min_time_sec"`
	MaxTime       float64         `json:"max_time_sec"`
	MeanTime      float64         `json:"mean_time_sec"`
	MedianTime    float64         `json:"median_time_sec"`
	StdDev        float64         `json:"std_dev_sec"`
	MinRTF        float64         `json:"min_rtf"`
	MaxRTF        float64         `json:"max_rtf"`
	MeanRTF       float64         `json:"mean_rtf"`
	MedianRTF     float64         `json:"median_rtf"`
	AllResults    []BenchmarkResult `json:"all_results"`
}

// BenchmarkReport holds results for all audio files
type BenchmarkReport struct {
	ModelPath   string            `json:"model_path"`
	ComputeType string            `json:"compute_type"`
	BeamSize    int               `json:"beam_size"`
	Threads     int               `json:"threads"`
	Language    string            `json:"language"`
	Task        string            `json:"task"`
	Timestamp   string            `json:"timestamp"`
	FileStats   []BenchmarkStats  `json:"file_stats"`
}

func main() {
	flag.Usage = usage
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model is required")
		fmt.Fprintln(os.Stderr, "")
		usage()
		os.Exit(1)
	}

	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: at least one audio file argument is required")
		fmt.Fprintln(os.Stderr, "")
		usage()
		os.Exit(1)
	}

	audioFiles := flag.Args()

	// Load model once
	fmt.Printf("Loading model from %s...\n", *modelPath)
	config := whisper.ModelConfig{
		Device:       "cpu",
		ComputeType:  *computeType,
		InterThreads: 1,
		IntraThreads: *threads,
	}

	start := time.Now()
	mdl, err := whisper.LoadModel(*modelPath, config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer mdl.Close()

	fmt.Printf("Model loaded in %v\n", time.Since(start))
	fmt.Printf("  Multilingual: %v\n", mdl.IsMultilingual())
	fmt.Printf("  Mel bins: %d\n", mdl.NumMels())
	fmt.Println()

	// Create benchmark report
	report := BenchmarkReport{
		ModelPath:   *modelPath,
		ComputeType: *computeType,
		BeamSize:    *beamSize,
		Threads:     *threads,
		Language:    *language,
		Task:        *task,
		Timestamp:   time.Now().Format(time.RFC3339),
		FileStats:   make([]BenchmarkStats, 0, len(audioFiles)),
	}

	// Benchmark each audio file
	for _, audioFile := range audioFiles {
		stats := benchmarkFile(mdl, audioFile)
		report.FileStats = append(report.FileStats, stats)
		printStats(stats)
	}

	// Save results if requested
	if *outputJSON != "" {
		if err := saveJSON(report, *outputJSON); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving JSON: %v\n", err)
		} else {
			fmt.Printf("\nResults saved to %s\n", *outputJSON)
		}
	}

	if *outputCSV != "" {
		if err := saveCSV(report, *outputCSV); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving CSV: %v\n", err)
		} else {
			fmt.Printf("Results saved to %s\n", *outputCSV)
		}
	}

	// Print summary
	fmt.Println("\n" + separator(80))
	fmt.Println("SUMMARY")
	fmt.Println(separator(80))

	var totalAudioDuration float64
	var totalTranscriptionTime float64

	for _, stats := range report.FileStats {
		totalAudioDuration += stats.AudioDuration
		totalTranscriptionTime += stats.MeanTime
	}

	overallRTF := totalAudioDuration / totalTranscriptionTime

	fmt.Printf("Total audio duration:       %.2fs\n", totalAudioDuration)
	fmt.Printf("Total transcription time:   %.2fs (mean)\n", totalTranscriptionTime)
	fmt.Printf("Overall RTF:                %.2fx\n", overallRTF)
	fmt.Printf("Files processed:            %d\n", len(audioFiles))
	fmt.Printf("Iterations per file:        %d\n", *iterations)
	fmt.Println(separator(80))
}

func benchmarkFile(mdl *whisper.Model, audioFile string) BenchmarkStats {
	fmt.Printf("Benchmarking: %s\n", filepath.Base(audioFile))
	fmt.Println(separator(80))

	results := make([]BenchmarkResult, 0, *iterations)

	// Warmup run (if enabled)
	if *warmup {
		if *verbose {
			fmt.Println("Running warmup iteration...")
		}
		_, err := mdl.TranscribeFile(audioFile,
			whisper.WithLanguage(*language),
			whisper.WithTask(*task),
			whisper.WithBeamSize(*beamSize),
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: warmup failed: %v\n", err)
		}
	}

	// Run benchmark iterations
	for i := 0; i < *iterations; i++ {
		if *verbose {
			fmt.Printf("Iteration %d/%d...\n", i+1, *iterations)
		}

		start := time.Now()
		result, err := mdl.TranscribeFile(audioFile,
			whisper.WithLanguage(*language),
			whisper.WithTask(*task),
			whisper.WithBeamSize(*beamSize),
		)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Fprintf(os.Stderr, "Error transcribing (iteration %d): %v\n", i+1, err)
			continue
		}

		audioDuration := result.Duration.Seconds()
		transcriptionSec := elapsed.Seconds()
		rtf := audioDuration / transcriptionSec

		benchResult := BenchmarkResult{
			AudioFile:         audioFile,
			AudioDuration:     audioDuration,
			TranscriptionTime: elapsed,
			TranscriptionSec:  transcriptionSec,
			RTF:               rtf,
			Language:          result.Language,
			SegmentCount:      len(result.Segments),
			TextLength:        len(result.Text),
		}

		results = append(results, benchResult)

		if *verbose {
			fmt.Printf("  Time: %.3fs | RTF: %.2fx | Lang: %s | Segments: %d\n",
				transcriptionSec, rtf, result.Language, len(result.Segments))
		}
	}

	if len(results) == 0 {
		fmt.Fprintf(os.Stderr, "Error: no successful iterations for %s\n", audioFile)
		return BenchmarkStats{AudioFile: audioFile, Iterations: 0}
	}

	// Calculate statistics
	stats := calculateStats(audioFile, results)

	return stats
}

func calculateStats(audioFile string, results []BenchmarkResult) BenchmarkStats {
	if len(results) == 0 {
		return BenchmarkStats{AudioFile: audioFile, Iterations: 0}
	}

	// Extract times and RTFs
	times := make([]float64, len(results))
	rtfs := make([]float64, len(results))

	for i, r := range results {
		times[i] = r.TranscriptionSec
		rtfs[i] = r.RTF
	}

	// Sort for median calculation
	sortedTimes := make([]float64, len(times))
	sortedRTFs := make([]float64, len(rtfs))
	copy(sortedTimes, times)
	copy(sortedRTFs, rtfs)
	sort.Float64s(sortedTimes)
	sort.Float64s(sortedRTFs)

	stats := BenchmarkStats{
		AudioFile:     audioFile,
		Iterations:    len(results),
		AudioDuration: results[0].AudioDuration,
		MinTime:       sortedTimes[0],
		MaxTime:       sortedTimes[len(sortedTimes)-1],
		MeanTime:      mean(times),
		MedianTime:    median(sortedTimes),
		StdDev:        stdDev(times),
		MinRTF:        sortedRTFs[0],
		MaxRTF:        sortedRTFs[len(sortedRTFs)-1],
		MeanRTF:       mean(rtfs),
		MedianRTF:     median(sortedRTFs),
		AllResults:    results,
	}

	return stats
}

func printStats(stats BenchmarkStats) {
	if stats.Iterations == 0 {
		fmt.Printf("No results for %s\n\n", stats.AudioFile)
		return
	}

	fmt.Println()
	fmt.Printf("Audio: %s\n", filepath.Base(stats.AudioFile))
	fmt.Printf("  Duration:           %.2fs\n", stats.AudioDuration)
	fmt.Printf("  Iterations:         %d\n", stats.Iterations)
	fmt.Println()
	fmt.Println("Transcription Time:")
	fmt.Printf("  Min:                %.3fs\n", stats.MinTime)
	fmt.Printf("  Max:                %.3fs\n", stats.MaxTime)
	fmt.Printf("  Mean:               %.3fs\n", stats.MeanTime)
	fmt.Printf("  Median:             %.3fs\n", stats.MedianTime)
	fmt.Printf("  Std Dev:            %.3fs\n", stats.StdDev)
	fmt.Println()
	fmt.Println("Real-Time Factor:")
	fmt.Printf("  Min RTF:            %.2fx (fastest)\n", stats.MinRTF)
	fmt.Printf("  Max RTF:            %.2fx (slowest)\n", stats.MaxRTF)
	fmt.Printf("  Mean RTF:           %.2fx\n", stats.MeanRTF)
	fmt.Printf("  Median RTF:         %.2fx\n", stats.MedianRTF)
	fmt.Println()
	fmt.Println("Transcription Info:")
	fmt.Printf("  Language:           %s\n", stats.AllResults[0].Language)
	fmt.Printf("  Segments:           %d\n", stats.AllResults[0].SegmentCount)
	fmt.Printf("  Text length:        %d chars\n", stats.AllResults[0].TextLength)
	fmt.Println(separator(80))
}

func saveJSON(report BenchmarkReport, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func saveCSV(report BenchmarkReport, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Write header
	header := []string{
		"audio_file",
		"iterations",
		"audio_duration_sec",
		"min_time_sec",
		"max_time_sec",
		"mean_time_sec",
		"median_time_sec",
		"std_dev_sec",
		"min_rtf",
		"max_rtf",
		"mean_rtf",
		"median_rtf",
		"language",
		"segments",
		"text_length",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	// Write data rows
	for _, stats := range report.FileStats {
		if stats.Iterations == 0 {
			continue
		}

		row := []string{
			filepath.Base(stats.AudioFile),
			fmt.Sprintf("%d", stats.Iterations),
			fmt.Sprintf("%.2f", stats.AudioDuration),
			fmt.Sprintf("%.3f", stats.MinTime),
			fmt.Sprintf("%.3f", stats.MaxTime),
			fmt.Sprintf("%.3f", stats.MeanTime),
			fmt.Sprintf("%.3f", stats.MedianTime),
			fmt.Sprintf("%.3f", stats.StdDev),
			fmt.Sprintf("%.2f", stats.MinRTF),
			fmt.Sprintf("%.2f", stats.MaxRTF),
			fmt.Sprintf("%.2f", stats.MeanRTF),
			fmt.Sprintf("%.2f", stats.MedianRTF),
			stats.AllResults[0].Language,
			fmt.Sprintf("%d", stats.AllResults[0].SegmentCount),
			fmt.Sprintf("%d", stats.AllResults[0].TextLength),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// Statistics helper functions
func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func median(sortedValues []float64) float64 {
	n := len(sortedValues)
	if n == 0 {
		return 0
	}
	if n%2 == 0 {
		return (sortedValues[n/2-1] + sortedValues[n/2]) / 2
	}
	return sortedValues[n/2]
}

func stdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - m
		sumSquares += diff * diff
	}
	return math.Sqrt(sumSquares / float64(len(values)))
}

func separator(width int) string {
	s := ""
	for i := 0; i < width; i++ {
		s += "="
	}
	return s
}

func usage() {
	fmt.Fprintf(os.Stderr, `whisper-benchmark - Benchmark transcription performance

Usage:
  whisper-benchmark [flags] <audio-file> [audio-file...]

Flags:
  --model string          Path to CTranslate2 model directory (required)
  --iterations int        Number of times to run each transcription (default 3)
  --language string       Language code or "auto" (default "auto")
  --task string           "transcribe" or "translate" (default "transcribe")
  --beam-size int         Beam search width (default 5)
  --compute-type string   Compute type: int8, float16, float32, default
  --threads int           Number of threads (0 = auto)
  --output-json string    Save results to JSON file
  --output-csv string     Save results to CSV file
  --warmup                Run one warmup iteration (default true)
  --verbose               Show detailed output for each iteration
  -h, --help              Show this help message

Examples:
  # Basic benchmark with 3 iterations (default)
  whisper-benchmark --model ./whisper-base-ct2 audio.wav

  # Benchmark multiple files with 5 iterations
  whisper-benchmark --model ./whisper-base-ct2 --iterations 5 audio1.wav audio2.wav

  # Save results to JSON and CSV
  whisper-benchmark --model ./whisper-base-ct2 --output-json results.json --output-csv results.csv audio.wav

  # Verbose output with custom settings
  whisper-benchmark --model ./whisper-base-ct2 --verbose --beam-size 10 --threads 4 audio.wav

Output:
  The benchmark reports several metrics:
  - Transcription time: min, max, mean, median, std dev
  - Real-Time Factor (RTF): audio_duration / transcription_time
    - Higher RTF = faster (e.g., 3.5x means 3.5 seconds of audio per second of processing)
  - Language detection results
  - Segment and text statistics
`)
}
