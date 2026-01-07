#ifndef WHISPER_CT2_H
#define WHISPER_CT2_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

typedef struct whisper_model* whisper_model_t;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

typedef enum {
    WHISPER_OK = 0,
    WHISPER_ERROR_INVALID_ARGUMENT = 1,
    WHISPER_ERROR_MODEL_NOT_FOUND = 2,
    WHISPER_ERROR_MODEL_LOAD_FAILED = 3,
    WHISPER_ERROR_AUDIO_LOAD_FAILED = 4,
    WHISPER_ERROR_TRANSCRIPTION_FAILED = 5,
    WHISPER_ERROR_OUT_OF_MEMORY = 6,
    WHISPER_ERROR_INVALID_MODEL = 7,
    WHISPER_ERROR_NOT_MULTILINGUAL = 8,
    WHISPER_ERROR_UNSUPPORTED_AUDIO_FORMAT = 9,
    WHISPER_ERROR_INTERNAL = 99
} whisper_error_t;

/* Get human-readable error message for an error code */
const char* whisper_error_message(whisper_error_t error);

/* Get last error for current thread (thread-local) */
whisper_error_t whisper_get_last_error(void);

/* Get detailed error message for last error (thread-local) */
const char* whisper_get_last_error_message(void);

/* Clear the last error state */
void whisper_clear_error(void);

/* ============================================================================
 * Model Configuration
 * ============================================================================ */

typedef struct {
    const char* device;           /* "cpu" only for now */
    const char* compute_type;     /* "int8", "int16", "float16", "float32", "default" */
    int32_t inter_threads;        /* Threads for batch parallelization (default: 1) */
    int32_t intra_threads;        /* Threads within operations (default: 0 = auto) */
} whisper_model_config_t;

/* Initialize config with defaults */
void whisper_model_config_init(whisper_model_config_t* config);

/* ============================================================================
 * Model Lifecycle
 * ============================================================================ */

/* Load a model from a directory path (CTranslate2 format)
 *
 * @param model_path  Path to the CTranslate2 model directory
 * @param config      Model configuration (can be NULL for defaults)
 * @param out_model   Output pointer to receive the model handle
 * @return            WHISPER_OK on success, error code otherwise
 */
whisper_error_t whisper_model_load(
    const char* model_path,
    const whisper_model_config_t* config,
    whisper_model_t* out_model
);

/* Free model resources */
void whisper_model_free(whisper_model_t model);

/* Check if model supports multiple languages */
bool whisper_model_is_multilingual(whisper_model_t model);

/* Get number of mel frequency bins (80 or 128) */
int32_t whisper_model_n_mels(whisper_model_t model);

/* ============================================================================
 * Transcription Options
 * ============================================================================ */

typedef struct {
    /* Decoding parameters */
    int32_t beam_size;              /* Beam search width (default: 5) */
    float patience;                 /* Beam search patience factor (default: 1.0) */
    float length_penalty;           /* Length penalty (default: 1.0) */
    float repetition_penalty;       /* Repetition penalty (default: 1.0) */
    int32_t no_repeat_ngram_size;   /* Prevent n-gram repetition (default: 0) */
    int32_t max_length;             /* Max tokens per segment (default: 448) */

    /* Sampling parameters (for sampling_topk > 1) */
    int32_t sampling_topk;          /* Top-k sampling (default: 1 = greedy/beam) */
    float sampling_temperature;     /* Sampling temperature (default: 1.0) */

    /* Timestamp parameters */
    int32_t max_initial_timestamp_index;  /* Max initial timestamp (default: 50) */
    bool suppress_blank;            /* Suppress blank at beginning (default: true) */

    /* Suppress tokens (NULL = use model defaults) */
    const int32_t* suppress_tokens;
    size_t suppress_tokens_count;

    /* Language settings */
    const char* language;           /* Language code, e.g., "en", or "auto" for detection */
    const char* task;               /* "transcribe" or "translate" */

    /* Output options */
    bool return_scores;             /* Return confidence scores */
    bool return_no_speech_prob;     /* Return no-speech probability */
    bool word_timestamps;           /* Compute word-level timestamps (not yet implemented) */

    /* Performance options */
    float no_speech_threshold;      /* Skip chunks with no_speech_prob > this (default: 0.6, 0.0 = disabled) */
    bool condition_on_previous_text; /* Use previous text as context (default: true) */
    float compression_ratio_threshold; /* Skip chunks with compression_ratio > this (default: 2.4, 0.0 = disabled) */
    float logprob_threshold;        /* Skip chunks with avg logprob < this (default: -1.0, 0.0 = disabled) */

    /* Temperature fallback (for retrying on failure) */
    const float* temperature_fallback; /* Array of temperatures to try (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) */
    size_t temperature_fallback_count;  /* Number of temperatures (default: 6) */
} whisper_transcribe_options_t;

/* Initialize options with defaults */
void whisper_transcribe_options_init(whisper_transcribe_options_t* options);

/* ============================================================================
 * Transcription Results
 * ============================================================================ */

typedef struct {
    char* text;                     /* Transcribed text (UTF-8), owned by result */
    float start_time;               /* Start time in seconds */
    float end_time;                 /* End time in seconds */
    float score;                    /* Confidence score (if requested) */
    float no_speech_prob;           /* No-speech probability (if requested) */
} whisper_segment_t;

typedef struct {
    char* detected_language;        /* Detected language code, owned by result */
    float language_probability;     /* Detection confidence */
    whisper_segment_t* segments;    /* Array of segments, owned by result */
    size_t segment_count;           /* Number of segments */
    float duration;                 /* Total audio duration in seconds */
} whisper_result_t;

/* Free result resources (segments, strings, etc.) */
void whisper_result_free(whisper_result_t* result);

/* ============================================================================
 * Transcription API
 * ============================================================================ */

/* Transcribe audio from file path
 *
 * Supported formats: WAV, FLAC, OGG, MP3 (via libsndfile)
 * Audio is automatically resampled to 16kHz mono.
 *
 * @param model       Model handle
 * @param audio_path  Path to audio file
 * @param options     Transcription options (can be NULL for defaults)
 * @param out_result  Output pointer to receive results
 * @return            WHISPER_OK on success, error code otherwise
 */
whisper_error_t whisper_transcribe_file(
    whisper_model_t model,
    const char* audio_path,
    const whisper_transcribe_options_t* options,
    whisper_result_t* out_result
);

/* Transcribe audio from raw PCM samples
 *
 * @param model        Model handle
 * @param samples      Float32 PCM samples, mono, 16kHz, normalized to [-1, 1]
 * @param sample_count Number of samples
 * @param options      Transcription options (can be NULL for defaults)
 * @param out_result   Output pointer to receive results
 * @return             WHISPER_OK on success, error code otherwise
 */
whisper_error_t whisper_transcribe_pcm(
    whisper_model_t model,
    const float* samples,
    size_t sample_count,
    const whisper_transcribe_options_t* options,
    whisper_result_t* out_result
);

/* ============================================================================
 * Language Detection
 * ============================================================================ */

typedef struct {
    char* language;                 /* Language code, owned by caller after return */
    float probability;              /* Detection probability */
} whisper_language_prob_t;

/* Detect language from audio file
 *
 * @param model      Model handle (must be multilingual)
 * @param audio_path Path to audio file
 * @param out_probs  Output array of language probabilities (caller frees)
 * @param out_count  Number of languages in output array
 * @return           WHISPER_OK on success, error code otherwise
 */
whisper_error_t whisper_detect_language_file(
    whisper_model_t model,
    const char* audio_path,
    whisper_language_prob_t** out_probs,
    size_t* out_count
);

/* Detect language from PCM samples
 *
 * @param model        Model handle (must be multilingual)
 * @param samples      Float32 PCM samples, mono, 16kHz
 * @param sample_count Number of samples
 * @param out_probs    Output array of language probabilities (caller frees)
 * @param out_count    Number of languages in output array
 * @return             WHISPER_OK on success, error code otherwise
 */
whisper_error_t whisper_detect_language_pcm(
    whisper_model_t model,
    const float* samples,
    size_t sample_count,
    whisper_language_prob_t** out_probs,
    size_t* out_count
);

/* Free language detection results */
void whisper_language_probs_free(whisper_language_prob_t* probs, size_t count);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/* Get library version string */
const char* whisper_ct2_version(void);

/* Get CTranslate2 version string */
const char* whisper_ct2_ctranslate2_version(void);

/* Get supported audio formats (comma-separated) */
const char* whisper_ct2_supported_audio_formats(void);

#ifdef __cplusplus
}
#endif

#endif /* WHISPER_CT2_H */
