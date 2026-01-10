package whisper

// TranscribeOptions configures transcription behavior.
type TranscribeOptions struct {
	// BeamSize specifies the beam search width.
	// Higher values may improve accuracy but are slower.
	// Default: 5
	BeamSize int

	// BestOf is the number of candidate sequences to generate.
	// This is critical for beam search performance (maps to num_hypotheses in CTranslate2).
	// Default: 5 (same as faster-whisper)
	BestOf int

	// Patience is the beam search patience factor.
	// Default: 1.0
	Patience float32

	// LengthPenalty penalizes longer sequences.
	// Default: 1.0
	LengthPenalty float32

	// RepetitionPenalty penalizes repeated tokens.
	// Default: 1.0
	RepetitionPenalty float32

	// NoRepeatNgramSize prevents n-gram repetition.
	// 0 means disabled.
	// Default: 0
	NoRepeatNgramSize int

	// MaxLength is the maximum tokens per segment.
	// Default: 448
	MaxLength int

	// SamplingTopK enables top-k sampling.
	// 1 means greedy/beam search.
	// Default: 1
	SamplingTopK int

	// SamplingTemperature controls randomness in sampling.
	// Only used when SamplingTopK > 1.
	// Default: 1.0
	SamplingTemperature float32

	// MaxInitialTimestampIndex limits the initial timestamp.
	// Default: 50
	MaxInitialTimestampIndex int

	// SuppressBlank suppresses blank tokens at the beginning.
	// Default: true
	SuppressBlank bool

	// SuppressTokens is a list of token IDs to suppress.
	// nil means use model defaults.
	SuppressTokens []int32

	// Language specifies the spoken language.
	// Use ISO 639-1 codes like "en", "fr", "de", etc.
	// Use "auto" for automatic language detection (multilingual models only).
	// Default: "auto"
	Language string

	// Task specifies the transcription task.
	// Options: "transcribe" (same language) or "translate" (to English).
	// Default: "transcribe"
	Task string

	// ReturnScores includes confidence scores in the result.
	// Default: false
	ReturnScores bool

	// ReturnNoSpeechProb includes no-speech probability in the result.
	// Default: false
	ReturnNoSpeechProb bool

	// WordTimestamps enables word-level timestamp computation.
	// Note: Not yet implemented.
	// Default: false
	WordTimestamps bool

	// NoSpeechThreshold skips chunks with no_speech_prob above this value.
	// This matches faster-whisper's VAD behavior for silent chunk filtering.
	// Set to 0.0 to disable (process all chunks).
	// Default: 0.6
	NoSpeechThreshold float32

	// ConditionOnPreviousText uses previous transcribed text as context.
	// This improves accuracy and consistency across segments.
	// Default: true
	ConditionOnPreviousText bool

	// CompressionRatioThreshold skips chunks with compression_ratio above this.
	// Detects repetitive/hallucinated text. Set to 0.0 to disable.
	// Default: 2.4
	CompressionRatioThreshold float32

	// LogprobThreshold skips chunks with average log probability below this.
	// Detects low-confidence transcriptions. Set to 0.0 to disable.
	// Default: -1.0
	LogprobThreshold float32

	// TemperatureFallback is a list of temperatures to try on quality failure.
	// If nil, uses default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	// Set to empty slice to disable fallback.
	TemperatureFallback []float32
}

// DefaultTranscribeOptions returns sensible defaults for transcription.
func DefaultTranscribeOptions() TranscribeOptions {
	return TranscribeOptions{
		BeamSize:                 5,
		BestOf:                   5,  // Match faster-whisper default
		Patience:                 1.0,
		LengthPenalty:            1.0,
		RepetitionPenalty:        1.0,
		NoRepeatNgramSize:        0,
		MaxLength:                448,
		SamplingTopK:             1,
		SamplingTemperature:      1.0,
		MaxInitialTimestampIndex: 50,
		SuppressBlank:            true,
		SuppressTokens:           nil,
		Language:                  "auto",
		Task:                      "transcribe",
		ReturnScores:              false,
		ReturnNoSpeechProb:        false,
		WordTimestamps:            false,
		NoSpeechThreshold:         0.6,   // Match faster-whisper default
		ConditionOnPreviousText:   true,
		CompressionRatioThreshold: 2.4,   // Match faster-whisper default
		LogprobThreshold:          -1.0,  // Match faster-whisper default
		TemperatureFallback:       nil,   // Use default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	}
}

// Option is a functional option for configuring transcription.
type Option func(*TranscribeOptions)

// WithLanguage sets the language for transcription.
//
// Use ISO 639-1 codes: "en", "fr", "de", "es", "zh", "ja", etc.
// Use "auto" for automatic language detection (multilingual models only).
func WithLanguage(lang string) Option {
	return func(o *TranscribeOptions) {
		o.Language = lang
	}
}

// WithTask sets the transcription task.
//
// Options:
//   - "transcribe": Transcribe speech to text in the original language
//   - "translate": Translate speech to English text
func WithTask(task string) Option {
	return func(o *TranscribeOptions) {
		o.Task = task
	}
}

// WithBeamSize sets the beam search width.
//
// Higher values may improve accuracy but increase computation time.
// Typical values: 1 (greedy), 5 (default), 10 (higher quality).
func WithBeamSize(size int) Option {
	return func(o *TranscribeOptions) {
		o.BeamSize = size
	}
}

// WithBestOf sets the number of candidate sequences to generate.
//
// This is critical for beam search performance. Should typically match BeamSize.
// Default: 5 (same as faster-whisper)
func WithBestOf(count int) Option {
	return func(o *TranscribeOptions) {
		o.BestOf = count
	}
}

// WithPatience sets the beam search patience factor.
func WithPatience(patience float32) Option {
	return func(o *TranscribeOptions) {
		o.Patience = patience
	}
}

// WithLengthPenalty sets the length penalty.
//
// Values > 1.0 favor longer sequences, < 1.0 favor shorter.
func WithLengthPenalty(penalty float32) Option {
	return func(o *TranscribeOptions) {
		o.LengthPenalty = penalty
	}
}

// WithRepetitionPenalty sets the repetition penalty.
//
// Values > 1.0 discourage repetition.
func WithRepetitionPenalty(penalty float32) Option {
	return func(o *TranscribeOptions) {
		o.RepetitionPenalty = penalty
	}
}

// WithNoRepeatNgramSize prevents n-gram repetition.
//
// Set to 0 to disable, or a positive value to prevent that size n-gram
// from being repeated.
func WithNoRepeatNgramSize(size int) Option {
	return func(o *TranscribeOptions) {
		o.NoRepeatNgramSize = size
	}
}

// WithMaxLength sets the maximum tokens per segment.
func WithMaxLength(length int) Option {
	return func(o *TranscribeOptions) {
		o.MaxLength = length
	}
}

// WithSampling enables sampling with the given top-k and temperature.
//
// When topK > 1, the model samples from the top-k tokens instead of
// using beam search. Temperature controls randomness (higher = more random).
func WithSampling(topK int, temperature float32) Option {
	return func(o *TranscribeOptions) {
		o.SamplingTopK = topK
		o.SamplingTemperature = temperature
	}
}

// WithSuppressBlank controls blank token suppression.
func WithSuppressBlank(suppress bool) Option {
	return func(o *TranscribeOptions) {
		o.SuppressBlank = suppress
	}
}

// WithSuppressTokens sets token IDs to suppress during generation.
func WithSuppressTokens(tokens []int32) Option {
	return func(o *TranscribeOptions) {
		o.SuppressTokens = tokens
	}
}

// WithScores enables confidence score output.
func WithScores(enabled bool) Option {
	return func(o *TranscribeOptions) {
		o.ReturnScores = enabled
	}
}

// WithNoSpeechProb enables no-speech probability output.
func WithNoSpeechProb(enabled bool) Option {
	return func(o *TranscribeOptions) {
		o.ReturnNoSpeechProb = enabled
	}
}

// WithWordTimestamps enables word-level timestamp computation.
//
// Note: This feature is not yet implemented.
func WithWordTimestamps(enabled bool) Option {
	return func(o *TranscribeOptions) {
		o.WordTimestamps = enabled
	}
}

// WithNoSpeechThreshold sets the threshold for skipping silent chunks.
//
// Chunks with no_speech_prob above this value are skipped, matching
// faster-whisper's VAD behavior. Set to 0.0 to process all chunks.
// Default: 0.6
func WithNoSpeechThreshold(threshold float32) Option {
	return func(o *TranscribeOptions) {
		o.NoSpeechThreshold = threshold
	}
}

// WithConditionOnPreviousText controls context conditioning.
//
// When enabled, previous transcribed text is used as context for
// better accuracy and consistency. Disable for independent segments.
// Default: true
func WithConditionOnPreviousText(enabled bool) Option {
	return func(o *TranscribeOptions) {
		o.ConditionOnPreviousText = enabled
	}
}

// WithCompressionRatioThreshold sets the threshold for skipping repetitive text.
//
// Chunks with compression ratio above this value are skipped or retried.
// Set to 0.0 to disable. Default: 2.4
func WithCompressionRatioThreshold(threshold float32) Option {
	return func(o *TranscribeOptions) {
		o.CompressionRatioThreshold = threshold
	}
}

// WithLogprobThreshold sets the threshold for skipping low-confidence text.
//
// Chunks with average log probability below this value are skipped or retried.
// Set to 0.0 to disable. Default: -1.0
func WithLogprobThreshold(threshold float32) Option {
	return func(o *TranscribeOptions) {
		o.LogprobThreshold = threshold
	}
}

// WithTemperatureFallback sets the temperature fallback sequence.
//
// When a chunk fails quality checks, it's retried with each temperature.
// Default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
// Set to empty slice to disable fallback (use temperature from sampling options).
func WithTemperatureFallback(temps []float32) Option {
	return func(o *TranscribeOptions) {
		o.TemperatureFallback = temps
	}
}
