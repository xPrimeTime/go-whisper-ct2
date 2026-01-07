package whisper

/*
#include <whisper_ct2.h>
#include <stdlib.h>
*/
import "C"
import (
	"strings"
	"time"
	"unsafe"
)

// TranscribeFile transcribes audio from a file path.
//
// Supported formats include WAV, FLAC, OGG, MP3, and others supported
// by libsndfile. Audio is automatically resampled to 16kHz mono.
//
// Example:
//
//	result, err := model.TranscribeFile("audio.wav",
//	    whisper.WithLanguage("en"),
//	    whisper.WithBeamSize(5),
//	)
//	if err != nil {
//	    return err
//	}
//	fmt.Println(result.Text)
func (m *Model) TranscribeFile(path string, opts ...Option) (*Result, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}

	options := DefaultTranscribeOptions()
	for _, opt := range opts {
		opt(&options)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cOpts := toCOptions(&options)
	defer freeCOptions(&cOpts)

	var cResult C.whisper_result_t
	errCode := C.whisper_transcribe_file(m.handle, cPath, &cOpts, &cResult)
	if errCode != C.WHISPER_OK {
		return nil, errorFromCode(errCode)
	}
	defer C.whisper_result_free(&cResult)

	return resultFromC(&cResult), nil
}

// TranscribePCM transcribes raw PCM audio samples.
//
// The samples must be float32, mono channel, 16kHz sample rate,
// normalized to the range [-1, 1].
//
// Example:
//
//	// Assuming you have audio samples from another source
//	result, err := model.TranscribePCM(samples, whisper.WithLanguage("en"))
func (m *Model) TranscribePCM(samples []float32, opts ...Option) (*Result, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}

	options := DefaultTranscribeOptions()
	for _, opt := range opts {
		opt(&options)
	}

	cOpts := toCOptions(&options)
	defer freeCOptions(&cOpts)

	var cResult C.whisper_result_t
	errCode := C.whisper_transcribe_pcm(
		m.handle,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
		&cOpts,
		&cResult,
	)
	if errCode != C.WHISPER_OK {
		return nil, errorFromCode(errCode)
	}
	defer C.whisper_result_free(&cResult)

	return resultFromC(&cResult), nil
}

// DetectLanguage detects the spoken language in an audio file.
//
// This method requires a multilingual model. For English-only models,
// it returns ErrNotMultilingual.
//
// Returns a slice of LanguageProb sorted by probability (highest first).
//
// Example:
//
//	probs, err := model.DetectLanguage("audio.wav")
//	if err != nil {
//	    return err
//	}
//	fmt.Println("Detected language:", probs[0].Language)
func (m *Model) DetectLanguage(path string) ([]LanguageProb, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}
	if !m.IsMultilingual() {
		return nil, ErrNotMultilingual
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cProbs *C.whisper_language_prob_t
	var count C.size_t

	errCode := C.whisper_detect_language_file(m.handle, cPath, &cProbs, &count)
	if errCode != C.WHISPER_OK {
		return nil, errorFromCode(errCode)
	}
	defer C.whisper_language_probs_free(cProbs, count)

	return languageProbsFromC(cProbs, int(count)), nil
}

// DetectLanguagePCM detects the spoken language from PCM samples.
//
// The samples must be float32, mono channel, 16kHz sample rate.
// Only the first 30 seconds are used for detection.
func (m *Model) DetectLanguagePCM(samples []float32) ([]LanguageProb, error) {
	if m.handle == nil {
		return nil, ErrInvalidModel
	}
	if !m.IsMultilingual() {
		return nil, ErrNotMultilingual
	}
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}

	var cProbs *C.whisper_language_prob_t
	var count C.size_t

	errCode := C.whisper_detect_language_pcm(
		m.handle,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
		&cProbs,
		&count,
	)
	if errCode != C.WHISPER_OK {
		return nil, errorFromCode(errCode)
	}
	defer C.whisper_language_probs_free(cProbs, count)

	return languageProbsFromC(cProbs, int(count)), nil
}

// toCOptions converts Go options to C options struct.
func toCOptions(opts *TranscribeOptions) C.whisper_transcribe_options_t {
	var cOpts C.whisper_transcribe_options_t
	C.whisper_transcribe_options_init(&cOpts)

	cOpts.beam_size = C.int32_t(opts.BeamSize)
	cOpts.patience = C.float(opts.Patience)
	cOpts.length_penalty = C.float(opts.LengthPenalty)
	cOpts.repetition_penalty = C.float(opts.RepetitionPenalty)
	cOpts.no_repeat_ngram_size = C.int32_t(opts.NoRepeatNgramSize)
	cOpts.max_length = C.int32_t(opts.MaxLength)
	cOpts.sampling_topk = C.int32_t(opts.SamplingTopK)
	cOpts.sampling_temperature = C.float(opts.SamplingTemperature)
	cOpts.max_initial_timestamp_index = C.int32_t(opts.MaxInitialTimestampIndex)
	cOpts.suppress_blank = C.bool(opts.SuppressBlank)
	cOpts.return_scores = C.bool(opts.ReturnScores)
	cOpts.return_no_speech_prob = C.bool(opts.ReturnNoSpeechProb)
	cOpts.word_timestamps = C.bool(opts.WordTimestamps)
	cOpts.no_speech_threshold = C.float(opts.NoSpeechThreshold)
	cOpts.condition_on_previous_text = C.bool(opts.ConditionOnPreviousText)
	cOpts.compression_ratio_threshold = C.float(opts.CompressionRatioThreshold)
	cOpts.logprob_threshold = C.float(opts.LogprobThreshold)

	if len(opts.TemperatureFallback) > 0 {
		cOpts.temperature_fallback = (*C.float)(unsafe.Pointer(&opts.TemperatureFallback[0]))
		cOpts.temperature_fallback_count = C.size_t(len(opts.TemperatureFallback))
	}

	if opts.Language != "" {
		cOpts.language = C.CString(opts.Language)
	}
	if opts.Task != "" {
		cOpts.task = C.CString(opts.Task)
	}

	if len(opts.SuppressTokens) > 0 {
		cOpts.suppress_tokens = (*C.int32_t)(unsafe.Pointer(&opts.SuppressTokens[0]))
		cOpts.suppress_tokens_count = C.size_t(len(opts.SuppressTokens))
	}

	return cOpts
}

// freeCOptions frees C strings allocated in toCOptions.
func freeCOptions(cOpts *C.whisper_transcribe_options_t) {
	if cOpts.language != nil {
		C.free(unsafe.Pointer(cOpts.language))
	}
	if cOpts.task != nil {
		C.free(unsafe.Pointer(cOpts.task))
	}
}

// resultFromC converts a C result to a Go Result.
func resultFromC(cResult *C.whisper_result_t) *Result {
	result := &Result{
		Language:            C.GoString(cResult.detected_language),
		LanguageProbability: float32(cResult.language_probability),
		Duration:            time.Duration(float64(cResult.duration) * float64(time.Second)),
	}

	segmentCount := int(cResult.segment_count)
	if segmentCount > 0 && cResult.segments != nil {
		cSegments := unsafe.Slice(cResult.segments, segmentCount)
		result.Segments = make([]Segment, segmentCount)

		var textBuilder strings.Builder
		for i, cs := range cSegments {
			text := C.GoString(cs.text)
			result.Segments[i] = Segment{
				Text:         text,
				Start:        time.Duration(float64(cs.start_time) * float64(time.Second)),
				End:          time.Duration(float64(cs.end_time) * float64(time.Second)),
				Score:        float32(cs.score),
				NoSpeechProb: float32(cs.no_speech_prob),
			}
			if i > 0 {
				textBuilder.WriteString(" ")
			}
			textBuilder.WriteString(text)
		}
		result.Text = strings.TrimSpace(textBuilder.String())
	}

	return result
}

// languageProbsFromC converts C language probabilities to Go slice.
func languageProbsFromC(cProbs *C.whisper_language_prob_t, count int) []LanguageProb {
	if count == 0 || cProbs == nil {
		return nil
	}

	probs := make([]LanguageProb, count)
	cProbsSlice := unsafe.Slice(cProbs, count)
	for i, cp := range cProbsSlice {
		probs[i] = LanguageProb{
			Language:    C.GoString(cp.language),
			Probability: float32(cp.probability),
		}
	}

	return probs
}
