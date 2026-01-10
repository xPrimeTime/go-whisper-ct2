#include "whisper_ct2.h"
#include "audio_processor.h"

#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/replica_pool.h>
#include <ctranslate2/vocabulary.h>

#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <future>
#include <fstream>
#include <zlib.h>

using namespace whisper_ct2;


// Version info
static const char* VERSION = "1.1.0";

// Internal model wrapper
struct whisper_model {
    std::unique_ptr<ctranslate2::models::Whisper> model;
    std::unique_ptr<ctranslate2::Vocabulary> vocabulary;
    bool is_multilingual;
    int n_mels;
    size_t eot_token_id;  // End of text token ID for filtering
};

// Thread-local error state
thread_local whisper_error_t g_last_error = WHISPER_OK;
thread_local std::string g_last_error_message;

// Helper: Set error state
static void set_error(whisper_error_t error, const std::string& message) {
    g_last_error = error;
    g_last_error_message = message;
}

// Helper: Clear error state
static void clear_error() {
    g_last_error = WHISPER_OK;
    g_last_error_message.clear();
}

// Helper: Duplicate string (caller must free)
static char* strdup_safe(const std::string& s) {
    char* result = static_cast<char*>(malloc(s.size() + 1));
    if (result) {
        std::strcpy(result, s.c_str());
    }
    return result;
}

// Helper: Decode token IDs to text (matching Python's tokenizer.decode())
// Filters out tokens >= EOT, decodes using vocabulary, cleans BPE artifacts
// OPTIMIZED: Use append() and in-place cleaning for better performance
static std::string decode_tokens(
    const std::vector<size_t>& token_ids,
    const ctranslate2::Vocabulary& vocab,
    size_t eot_token_id
) {
    std::string result;
    result.reserve(token_ids.size() * 4);

    // Decode tokens directly - use append() which is faster than operator+=
    for (size_t token_id : token_ids) {
        if (token_id >= eot_token_id) {
            continue;
        }
        if (token_id < vocab.size()) {
            const std::string& token = vocab.to_token(token_id);
            result.append(token);
        }
    }

    // Clean BPE artifacts in-place to avoid extra allocation
    size_t write_pos = 0;
    for (size_t read_pos = 0; read_pos < result.size(); ++read_pos, ++write_pos) {
        if (read_pos + 1 < result.size() &&
            static_cast<unsigned char>(result[read_pos]) == 0xC4 &&
            static_cast<unsigned char>(result[read_pos + 1]) == 0xA0) {
            result[write_pos] = ' ';
            ++read_pos;
        } else {
            if (write_pos != read_pos) {
                result[write_pos] = result[read_pos];
            }
        }
    }
    result.resize(write_pos);

    return result;
}

// Helper: Clean BPE artifacts from text
// Replaces Ġ (U+0120) with space, handles other BPE markers
static std::string clean_text(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        // Check for Ġ which is UTF-8 encoded as 0xC4 0xA0
        if (i + 1 < text.size() &&
            static_cast<unsigned char>(text[i]) == 0xC4 &&
            static_cast<unsigned char>(text[i + 1]) == 0xA0) {
            result += ' ';
            ++i;  // Skip second byte
        } else {
            result += text[i];
        }
    }

    return result;
}

// Helper: Extract language code from token like "<|en|>"
static std::string extract_language_code(const std::string& token) {
    if (token.size() > 4 && token[0] == '<' && token[1] == '|' &&
        token[token.size()-2] == '|' && token[token.size()-1] == '>') {
        return token.substr(2, token.size() - 4);
    }
    return token;
}

// Helper: Convert compute type string to enum
static ctranslate2::ComputeType parse_compute_type(const char* compute_type) {
    if (!compute_type || std::string(compute_type) == "default") {
        return ctranslate2::ComputeType::DEFAULT;
    }
    std::string ct = compute_type;
    if (ct == "int8") return ctranslate2::ComputeType::INT8;
    if (ct == "int8_float32") return ctranslate2::ComputeType::INT8_FLOAT32;
    if (ct == "int8_float16") return ctranslate2::ComputeType::INT8_FLOAT16;
    if (ct == "int8_bfloat16") return ctranslate2::ComputeType::INT8_BFLOAT16;
    if (ct == "int16") return ctranslate2::ComputeType::INT16;
    if (ct == "float16") return ctranslate2::ComputeType::FLOAT16;
    if (ct == "bfloat16") return ctranslate2::ComputeType::BFLOAT16;
    if (ct == "float32") return ctranslate2::ComputeType::FLOAT32;
    return ctranslate2::ComputeType::DEFAULT;
}

// Helper: Build prompt tokens for Whisper
static std::vector<std::string> build_prompt(
    const char* language,
    const char* task,
    bool is_multilingual,
    const std::string& previous_text = ""
) {
    std::vector<std::string> prompt;
    prompt.push_back("<|startoftranscript|>");

    if (is_multilingual && language && std::string(language) != "auto") {
        prompt.push_back("<|" + std::string(language) + "|>");
    }

    if (task && std::string(task) == "translate") {
        prompt.push_back("<|translate|>");
    } else {
        prompt.push_back("<|transcribe|>");
    }

    // Don't add <|notimestamps|> so we get segment timestamps

    // Add previous text for context if provided
    if (!previous_text.empty()) {
        prompt.push_back(previous_text);
    }

    return prompt;
}

// Helper: Calculate compression ratio for text (matches faster-whisper)
static float calculate_compression_ratio(const std::string& text) {
    if (text.empty()) return 1.0f;

    // Use zlib compression ratio like faster-whisper does
    // compression_ratio = len(text) / len(zlib.compress(text))
    uLongf compressed_size = compressBound(text.length());
    std::vector<Bytef> compressed(compressed_size);

    int result = compress(compressed.data(), &compressed_size,
                         reinterpret_cast<const Bytef*>(text.data()), text.length());

    if (result != Z_OK) {
        return 1.0f;  // Compression failed, assume no repetition
    }

    return static_cast<float>(text.length()) / static_cast<float>(compressed_size);
}

// Helper: Calculate average log probability
static float calculate_avg_logprob(const std::vector<float>& scores) {
    if (scores.empty()) return 0.0f;

    float sum = 0.0f;
    for (float score : scores) {
        sum += score;
    }
    return sum / scores.size();
}

// Helper: Extract text and timestamps from generation result (using token IDs)
// This matches Python's faster-whisper approach exactly
static void process_generation_result(
    const ctranslate2::models::WhisperGenerationResult& gen_result,
    const ctranslate2::Vocabulary& vocab,
    size_t eot_token_id,
    float time_offset,
    std::vector<whisper_segment_t>& segments,
    std::string& full_text,
    bool return_scores,
    bool return_no_speech_prob
) {
    // Process sequences (only use the first/best sequence when num_hypotheses > 1)
    for (size_t seq_idx = 0; seq_idx < gen_result.sequences_ids.size() && seq_idx < 1; ++seq_idx) {
        const auto& token_ids = gen_result.sequences_ids[seq_idx];

        // Parse token IDs to extract text and timestamps
        std::vector<size_t> text_tokens;
        float start_time = time_offset;
        float end_time = time_offset + 30.0f;  // Default to chunk end
        bool in_timestamp = false;

        for (size_t token_id : token_ids) {
            // Get the string representation for timestamp detection
            const std::string& token = vocab.to_token(token_id);

            // Check for timestamp tokens: <|0.00|>, <|0.02|>, etc.
            if (token.size() > 4 && token[0] == '<' && token[1] == '|' &&
                token[token.size()-2] == '|' && token[token.size()-1] == '>') {
                // Try to parse as timestamp
                std::string inner = token.substr(2, token.size() - 4);
                try {
                    float ts = std::stof(inner);
                    if (!in_timestamp) {
                        start_time = time_offset + ts;
                        in_timestamp = true;
                    } else {
                        end_time = time_offset + ts;
                    }
                } catch (...) {
                    // Not a timestamp, might be special token
                }
            } else if (token[0] != '<') {
                // Regular text token - collect ID for decoding
                text_tokens.push_back(token_id);
            }
        }

        if (!text_tokens.empty()) {
            // Decode tokens like Python does: filter EOT, decode, clean BPE
            std::string segment_text = decode_tokens(text_tokens, vocab, eot_token_id);

            // Trim whitespace
            while (!segment_text.empty() && std::isspace(segment_text.front())) {
                segment_text.erase(0, 1);
            }
            while (!segment_text.empty() && std::isspace(segment_text.back())) {
                segment_text.pop_back();
            }

            if (!segment_text.empty()) {
                whisper_segment_t seg;
                seg.text = strdup_safe(segment_text);
                seg.start_time = start_time;
                seg.end_time = end_time;
                seg.score = return_scores && seq_idx < gen_result.scores.size()
                            ? gen_result.scores[seq_idx] : 0.0f;
                seg.no_speech_prob = return_no_speech_prob ? gen_result.no_speech_prob : 0.0f;
                segments.push_back(seg);

                if (!full_text.empty()) {
                    full_text += " ";
                }
                full_text += segment_text;
            }
        }
    }
}

// Helper: Transcribe audio samples (core implementation)
static whisper_error_t transcribe_samples(
    whisper_model_t model,
    const std::vector<float>& samples,
    const whisper_transcribe_options_t* options,
    whisper_result_t* out_result
) {
    clear_error();

    // Use default options if not provided
    whisper_transcribe_options_t opts;
    if (options) {
        opts = *options;
    } else {
        whisper_transcribe_options_init(&opts);
    }

    try {
        // Compute mel spectrogram
        auto mel = AudioProcessor::compute_mel_spectrogram(samples, model->n_mels);
        const float duration = AudioProcessor::get_duration(samples.size());

        // Process in chunks
        std::vector<whisper_segment_t> all_segments;
        std::string full_text;

        const int total_frames = mel.empty() ? 0 : static_cast<int>(mel[0].size());
        const int chunk_frames = AudioProcessor::N_FRAMES_PER_CHUNK;
        float time_offset = 0.0f;

        // Build generation options
        ctranslate2::models::WhisperOptions gen_opts;
        gen_opts.beam_size = opts.beam_size;
        // Note: num_hypotheses is set dynamically in the temperature fallback loop
        gen_opts.patience = opts.patience;
        gen_opts.length_penalty = opts.length_penalty;
        gen_opts.repetition_penalty = opts.repetition_penalty;
        gen_opts.no_repeat_ngram_size = opts.no_repeat_ngram_size;
        gen_opts.max_length = opts.max_length;
        gen_opts.sampling_topk = opts.sampling_topk;
        gen_opts.sampling_temperature = opts.sampling_temperature;
        gen_opts.suppress_blank = opts.suppress_blank;
        gen_opts.return_scores = opts.return_scores;
        gen_opts.max_initial_timestamp_index = opts.max_initial_timestamp_index;
        // Auto-enable no_speech_prob if using threshold
        gen_opts.return_no_speech_prob = opts.return_no_speech_prob || (opts.no_speech_threshold > 0.0f);

        if (opts.suppress_tokens && opts.suppress_tokens_count > 0) {
            gen_opts.suppress_tokens.assign(
                opts.suppress_tokens,
                opts.suppress_tokens + opts.suppress_tokens_count
            );
        }

        // Detect language if needed
        std::string detected_language;
        float language_prob = 0.0f;

        // Track previous text for conditioning
        std::string previous_text;

        // Process chunks
        for (int start = 0; start < total_frames || start == 0; start += chunk_frames) {
            // Extract chunk
            int end = std::min(start + chunk_frames, total_frames);
            std::vector<std::vector<float>> chunk(model->n_mels);

            if (total_frames > 0) {
                for (int m = 0; m < model->n_mels; ++m) {
                    chunk[m].assign(mel[m].begin() + start, mel[m].begin() + end);
                }
                // Pad to chunk length
                chunk = AudioProcessor::pad_or_trim(chunk, chunk_frames);
            } else {
                // Empty audio - create zero-filled chunk
                for (int m = 0; m < model->n_mels; ++m) {
                    chunk[m].resize(chunk_frames, 0.0f);
                }
            }

            // Flatten and create StorageView
            auto flat = AudioProcessor::flatten(chunk);
            ctranslate2::Shape shape = {1, static_cast<ctranslate2::dim_t>(model->n_mels),
                                        static_cast<ctranslate2::dim_t>(chunk_frames)};
            ctranslate2::StorageView features(shape, flat, ctranslate2::Device::CPU);

            // Detect language on first chunk if auto
            if (start == 0 && model->is_multilingual &&
                (!opts.language || std::string(opts.language) == "auto")) {
                auto lang_futures = model->model->detect_language(features);
                if (!lang_futures.empty()) {
                    auto lang_results = lang_futures[0].get();
                    if (!lang_results.empty()) {
                        detected_language = extract_language_code(lang_results[0].first);
                        language_prob = lang_results[0].second;
                    }
                }
            } else if (opts.language && std::string(opts.language) != "auto") {
                detected_language = opts.language;
                language_prob = 1.0f;
            }

            // Build prompt with previous text context if enabled
            auto prompt = build_prompt(
                detected_language.empty() ? nullptr : detected_language.c_str(),
                opts.task,
                model->is_multilingual,
                opts.condition_on_previous_text ? previous_text : ""
            );

            // Default temperature fallback if not specified
            static const float default_temps[] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
            const float* temps = opts.temperature_fallback ? opts.temperature_fallback : default_temps;
            size_t temp_count = opts.temperature_fallback_count > 0 ? opts.temperature_fallback_count : 6;

            // Pre-encode features ONCE for all temperature attempts (critical for performance!)
            // This matches faster-whisper's behavior
            auto encoder_output_future = model->model->encode(features, /* to_cpu */ false);
            auto encoder_output = encoder_output_future.get();

            // Try generation with temperature fallback
            ctranslate2::models::WhisperGenerationResult best_result;
            bool success = false;

            // Track best fallback result if all temperatures fail (like faster-whisper)
            float best_fallback_logprob = -std::numeric_limits<float>::infinity();
            ctranslate2::models::WhisperGenerationResult best_fallback_result;
            bool has_fallback = false;
            bool has_cr_passed_fallback = false;

            for (size_t t = 0; t < temp_count && !success; ++t) {
                // Update temperature and beam search parameters
                auto temp_opts = gen_opts;
                temp_opts.sampling_temperature = temps[t];

                // Force return_scores=true for quality checking (like faster-whisper)
                temp_opts.return_scores = true;

                // Match faster-whisper's behavior:
                // - temperature == 0: beam_size=N, no num_hypotheses (greedy beam search)
                // - temperature > 0: beam_size=1, num_hypotheses=best_of (sampling)
                if (temps[t] > 0.0f) {
                    temp_opts.beam_size = 1;
                    temp_opts.num_hypotheses = opts.best_of;
                    temp_opts.sampling_topk = 0;  // Use all tokens for sampling
                } else {
                    temp_opts.beam_size = opts.beam_size;
                    temp_opts.num_hypotheses = 1;  // Only return best sequence for greedy
                }

                // Generate from pre-encoded features
                auto result_futures = model->model->generate(
                    encoder_output,
                    {prompt},
                    temp_opts
                );

                if (result_futures.empty()) continue;
                auto result = result_futures[0].get();

                // Extract text to check quality (using token IDs like Python)
                std::string chunk_text;
                if (!result.sequences_ids.empty()) {
                    const auto& token_ids = result.sequences_ids[0];
                    chunk_text = decode_tokens(token_ids, *model->vocabulary, model->eot_token_id);
                }

                // Calculate quality metrics (like faster-whisper lines 1465-1469)
                float compression_ratio = calculate_compression_ratio(chunk_text);
                float avg_logprob = calculate_avg_logprob(result.scores);

                // Check quality thresholds (like faster-whisper lines 1479-1505)
                bool needs_fallback = false;

                // Check compression ratio threshold
                bool passed_compression = opts.compression_ratio_threshold <= 0.0f ||
                                         compression_ratio <= opts.compression_ratio_threshold;

                if (!passed_compression) {
                    needs_fallback = true;  // too repetitive
                }

                // Check log probability threshold
                if (opts.logprob_threshold < 0.0f &&
                    avg_logprob < opts.logprob_threshold) {
                    needs_fallback = true;  // average log probability is too low
                }

                // Check if it's silence: if no_speech AND low quality, accept as silence
                // (like faster-whisper lines 1507-1513)
                if (opts.no_speech_threshold > 0.0f &&
                    result.no_speech_prob > opts.no_speech_threshold &&
                    opts.logprob_threshold < 0.0f &&
                    avg_logprob < opts.logprob_threshold) {
                    needs_fallback = false;  // silence - accept it
                }

                if (!needs_fallback) {
                    best_result = std::move(result);
                    success = true;
                } else {
                    // Track best fallback result (highest avg_logprob, prefer passed compression)
                    bool update_fallback = false;

                    if (!has_fallback) {
                        // No fallback yet, use this one
                        update_fallback = true;
                    } else if (passed_compression && !has_cr_passed_fallback) {
                        // This passed compression and previous didn't, prefer this
                        update_fallback = true;
                    } else if (passed_compression == has_cr_passed_fallback && avg_logprob > best_fallback_logprob) {
                        // Both same compression status, pick higher logprob
                        update_fallback = true;
                    }

                    if (update_fallback) {
                        best_fallback_result = result;  // Copy for fallback
                        best_fallback_logprob = avg_logprob;
                        has_fallback = true;
                        has_cr_passed_fallback = passed_compression;
                    }
                }
            }

            // If all temperatures failed, use best fallback (like faster-whisper lines 1518-1528)
            if (!success && has_fallback) {
                best_result = std::move(best_fallback_result);
                success = true;
            }

            // Process the best result
            if (success) {
                size_t segments_before = all_segments.size();
                process_generation_result(
                    best_result,
                    *model->vocabulary,
                    model->eot_token_id,
                    time_offset,
                    all_segments,
                    full_text,
                    opts.return_scores,
                    opts.return_no_speech_prob
                );

                // Update previous text with new segments for context
                if (opts.condition_on_previous_text) {
                    // Collect text from newly added segments
                    std::string chunk_text;
                    for (size_t i = segments_before; i < all_segments.size(); ++i) {
                        if (!chunk_text.empty()) chunk_text += " ";
                        chunk_text += all_segments[i].text;
                    }
                    // Keep last ~244 tokens worth of context (approximation)
                    if (!chunk_text.empty()) {
                        previous_text = chunk_text;
                        if (previous_text.length() > 500) {
                            previous_text = previous_text.substr(previous_text.length() - 500);
                        }
                    }
                }
            }

            time_offset += AudioProcessor::CHUNK_LENGTH;

            // Break if we've processed all frames
            if (end >= total_frames) {
                break;
            }
        }

        // Populate result
        out_result->detected_language = strdup_safe(detected_language);
        out_result->language_probability = language_prob;
        out_result->duration = duration;
        out_result->segment_count = all_segments.size();

        if (!all_segments.empty()) {
            out_result->segments = static_cast<whisper_segment_t*>(
                malloc(all_segments.size() * sizeof(whisper_segment_t))
            );
            if (!out_result->segments) {
                set_error(WHISPER_ERROR_OUT_OF_MEMORY, "Failed to allocate segments");
                return WHISPER_ERROR_OUT_OF_MEMORY;
            }
            std::copy(all_segments.begin(), all_segments.end(), out_result->segments);
        } else {
            out_result->segments = nullptr;
        }

        return WHISPER_OK;

    } catch (const std::exception& e) {
        set_error(WHISPER_ERROR_TRANSCRIPTION_FAILED, e.what());
        return WHISPER_ERROR_TRANSCRIPTION_FAILED;
    }
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

const char* whisper_error_message(whisper_error_t error) {
    switch (error) {
        case WHISPER_OK: return "Success";
        case WHISPER_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case WHISPER_ERROR_MODEL_NOT_FOUND: return "Model not found";
        case WHISPER_ERROR_MODEL_LOAD_FAILED: return "Failed to load model";
        case WHISPER_ERROR_AUDIO_LOAD_FAILED: return "Failed to load audio";
        case WHISPER_ERROR_TRANSCRIPTION_FAILED: return "Transcription failed";
        case WHISPER_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case WHISPER_ERROR_INVALID_MODEL: return "Invalid model handle";
        case WHISPER_ERROR_NOT_MULTILINGUAL: return "Model is not multilingual";
        case WHISPER_ERROR_UNSUPPORTED_AUDIO_FORMAT: return "Unsupported audio format";
        case WHISPER_ERROR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

whisper_error_t whisper_get_last_error(void) {
    return g_last_error;
}

const char* whisper_get_last_error_message(void) {
    return g_last_error_message.c_str();
}

void whisper_clear_error(void) {
    clear_error();
}

void whisper_model_config_init(whisper_model_config_t* config) {
    if (config) {
        config->device = "cpu";
        config->compute_type = "default";
        config->inter_threads = 1;
        config->intra_threads = 0;  // Auto
    }
}

whisper_error_t whisper_model_load(
    const char* model_path,
    const whisper_model_config_t* config,
    whisper_model_t* out_model
) {
    clear_error();

    if (!model_path || !out_model) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Null argument");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto model = std::make_unique<whisper_model>();

        // Parse configuration
        ctranslate2::ComputeType compute_type = ctranslate2::ComputeType::DEFAULT;
        size_t inter_threads = 1;
        size_t intra_threads = 0;

        if (config) {
            compute_type = parse_compute_type(config->compute_type);
            inter_threads = config->inter_threads > 0 ? config->inter_threads : 1;
            intra_threads = config->intra_threads >= 0 ? config->intra_threads : 0;
        }

        // Load model with ReplicaPoolConfig API
        // Note: Python's intra_threads maps to num_threads_per_replica
        //       Python's inter_threads maps to max_queued_batches
        ctranslate2::ReplicaPoolConfig pool_config;
        pool_config.num_threads_per_replica = intra_threads;
        pool_config.max_queued_batches = inter_threads;

        model->model = std::make_unique<ctranslate2::models::Whisper>(
            model_path,
            ctranslate2::Device::CPU,
            compute_type,
            std::vector<int>{0},  // device_indices
            false,  // tensor_parallel
            pool_config
        );

        model->is_multilingual = model->model->is_multilingual();
        model->n_mels = model->model->n_mels();

        // Load vocabulary from model directory (Python-free!)
        std::string vocab_file_path = std::string(model_path) + "/vocabulary.txt";
        std::ifstream vocab_file(vocab_file_path);
        if (!vocab_file.is_open()) {
            set_error(WHISPER_ERROR_MODEL_LOAD_FAILED,
                     "Failed to load vocabulary file: " + vocab_file_path);
            return WHISPER_ERROR_MODEL_LOAD_FAILED;
        }

        model->vocabulary = std::make_unique<ctranslate2::Vocabulary>(
            ctranslate2::Vocabulary::from_text_file(vocab_file)
        );
        vocab_file.close();

        // Store EOT token ID for filtering (matching Python's approach)
        // Whisper uses token 50257 as EOT (<|endoftext|>)
        model->eot_token_id = 50257;

        *out_model = model.release();
        return WHISPER_OK;

    } catch (const std::exception& e) {
        set_error(WHISPER_ERROR_MODEL_LOAD_FAILED, e.what());
        return WHISPER_ERROR_MODEL_LOAD_FAILED;
    }
}

void whisper_model_free(whisper_model_t model) {
    delete model;
}

bool whisper_model_is_multilingual(whisper_model_t model) {
    return model ? model->is_multilingual : false;
}

int32_t whisper_model_n_mels(whisper_model_t model) {
    return model ? model->n_mels : 0;
}

void whisper_transcribe_options_init(whisper_transcribe_options_t* options) {
    if (options) {
        options->beam_size = 5;
        options->best_of = 5;  // Match faster-whisper default
        options->patience = 1.0f;
        options->length_penalty = 1.0f;
        options->repetition_penalty = 1.0f;
        options->no_repeat_ngram_size = 0;
        options->max_length = 448;
        options->sampling_topk = 1;
        options->sampling_temperature = 1.0f;
        options->max_initial_timestamp_index = 50;
        options->suppress_blank = true;
        options->suppress_tokens = nullptr;
        options->suppress_tokens_count = 0;
        options->language = "auto";
        options->task = "transcribe";
        options->return_scores = false;
        options->return_no_speech_prob = false;
        options->word_timestamps = false;
        options->no_speech_threshold = 0.6f;  // Match faster-whisper default
        options->condition_on_previous_text = true;
        options->compression_ratio_threshold = 2.4f;  // Match faster-whisper default
        options->logprob_threshold = -1.0f;  // Match faster-whisper default
        options->temperature_fallback = nullptr;  // Use default array if nullptr
        options->temperature_fallback_count = 0;
    }
}

void whisper_result_free(whisper_result_t* result) {
    if (result) {
        free(result->detected_language);
        result->detected_language = nullptr;

        if (result->segments) {
            for (size_t i = 0; i < result->segment_count; ++i) {
                free(result->segments[i].text);
            }
            free(result->segments);
            result->segments = nullptr;
        }
        result->segment_count = 0;
    }
}

whisper_error_t whisper_transcribe_file(
    whisper_model_t model,
    const char* audio_path,
    const whisper_transcribe_options_t* options,
    whisper_result_t* out_result
) {
    clear_error();

    if (!model) {
        set_error(WHISPER_ERROR_INVALID_MODEL, "Null model handle");
        return WHISPER_ERROR_INVALID_MODEL;
    }
    if (!audio_path || !out_result) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Null argument");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }

    // Initialize result
    std::memset(out_result, 0, sizeof(*out_result));

    try {
        // Load audio
        auto samples = AudioProcessor::load_audio(audio_path);
        return transcribe_samples(model, samples, options, out_result);

    } catch (const std::exception& e) {
        set_error(WHISPER_ERROR_AUDIO_LOAD_FAILED, e.what());
        return WHISPER_ERROR_AUDIO_LOAD_FAILED;
    }
}

whisper_error_t whisper_transcribe_pcm(
    whisper_model_t model,
    const float* samples,
    size_t sample_count,
    const whisper_transcribe_options_t* options,
    whisper_result_t* out_result
) {
    clear_error();

    if (!model) {
        set_error(WHISPER_ERROR_INVALID_MODEL, "Null model handle");
        return WHISPER_ERROR_INVALID_MODEL;
    }
    if (!samples || !out_result) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Null argument");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }
    if (sample_count == 0) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Empty audio");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }

    // Initialize result
    std::memset(out_result, 0, sizeof(*out_result));

    // Copy samples and transcribe
    std::vector<float> audio(samples, samples + sample_count);
    return transcribe_samples(model, audio, options, out_result);
}

whisper_error_t whisper_detect_language_file(
    whisper_model_t model,
    const char* audio_path,
    whisper_language_prob_t** out_probs,
    size_t* out_count
) {
    clear_error();

    if (!model) {
        set_error(WHISPER_ERROR_INVALID_MODEL, "Null model handle");
        return WHISPER_ERROR_INVALID_MODEL;
    }
    if (!audio_path || !out_probs || !out_count) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Null argument");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }
    if (!model->is_multilingual) {
        set_error(WHISPER_ERROR_NOT_MULTILINGUAL, "Model is not multilingual");
        return WHISPER_ERROR_NOT_MULTILINGUAL;
    }

    try {
        // Load audio
        auto samples = AudioProcessor::load_audio(audio_path);

        // Compute mel spectrogram (first 30 seconds only)
        if (samples.size() > static_cast<size_t>(AudioProcessor::N_SAMPLES_PER_CHUNK)) {
            samples.resize(AudioProcessor::N_SAMPLES_PER_CHUNK);
        }
        auto mel = AudioProcessor::compute_mel_spectrogram(samples, model->n_mels);
        mel = AudioProcessor::pad_or_trim(mel, AudioProcessor::N_FRAMES_PER_CHUNK);

        // Create StorageView
        auto flat = AudioProcessor::flatten(mel);
        ctranslate2::Shape shape = {1, static_cast<ctranslate2::dim_t>(model->n_mels),
                                    static_cast<ctranslate2::dim_t>(AudioProcessor::N_FRAMES_PER_CHUNK)};
        ctranslate2::StorageView features(shape, flat, ctranslate2::Device::CPU);

        // Detect language
        auto lang_futures = model->model->detect_language(features);

        if (lang_futures.empty()) {
            *out_probs = nullptr;
            *out_count = 0;
            return WHISPER_OK;
        }

        auto probs = lang_futures[0].get();
        if (probs.empty()) {
            *out_probs = nullptr;
            *out_count = 0;
            return WHISPER_OK;
        }
        *out_count = probs.size();
        *out_probs = static_cast<whisper_language_prob_t*>(
            malloc(probs.size() * sizeof(whisper_language_prob_t))
        );

        if (!*out_probs) {
            set_error(WHISPER_ERROR_OUT_OF_MEMORY, "Failed to allocate language probabilities");
            return WHISPER_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < probs.size(); ++i) {
            (*out_probs)[i].language = strdup_safe(extract_language_code(probs[i].first));
            (*out_probs)[i].probability = probs[i].second;
        }

        return WHISPER_OK;

    } catch (const std::exception& e) {
        set_error(WHISPER_ERROR_AUDIO_LOAD_FAILED, e.what());
        return WHISPER_ERROR_AUDIO_LOAD_FAILED;
    }
}

whisper_error_t whisper_detect_language_pcm(
    whisper_model_t model,
    const float* samples,
    size_t sample_count,
    whisper_language_prob_t** out_probs,
    size_t* out_count
) {
    clear_error();

    if (!model) {
        set_error(WHISPER_ERROR_INVALID_MODEL, "Null model handle");
        return WHISPER_ERROR_INVALID_MODEL;
    }
    if (!samples || !out_probs || !out_count) {
        set_error(WHISPER_ERROR_INVALID_ARGUMENT, "Null argument");
        return WHISPER_ERROR_INVALID_ARGUMENT;
    }
    if (!model->is_multilingual) {
        set_error(WHISPER_ERROR_NOT_MULTILINGUAL, "Model is not multilingual");
        return WHISPER_ERROR_NOT_MULTILINGUAL;
    }

    try {
        // Copy samples (limit to first 30 seconds)
        size_t max_samples = static_cast<size_t>(AudioProcessor::N_SAMPLES_PER_CHUNK);
        size_t actual_samples = std::min(sample_count, max_samples);
        std::vector<float> audio(samples, samples + actual_samples);

        // Compute mel spectrogram
        auto mel = AudioProcessor::compute_mel_spectrogram(audio, model->n_mels);
        mel = AudioProcessor::pad_or_trim(mel, AudioProcessor::N_FRAMES_PER_CHUNK);

        // Create StorageView
        auto flat = AudioProcessor::flatten(mel);
        ctranslate2::Shape shape = {1, static_cast<ctranslate2::dim_t>(model->n_mels),
                                    static_cast<ctranslate2::dim_t>(AudioProcessor::N_FRAMES_PER_CHUNK)};
        ctranslate2::StorageView features(shape, flat, ctranslate2::Device::CPU);

        // Detect language
        auto lang_futures = model->model->detect_language(features);

        if (lang_futures.empty()) {
            *out_probs = nullptr;
            *out_count = 0;
            return WHISPER_OK;
        }

        auto probs = lang_futures[0].get();
        if (probs.empty()) {
            *out_probs = nullptr;
            *out_count = 0;
            return WHISPER_OK;
        }
        *out_count = probs.size();
        *out_probs = static_cast<whisper_language_prob_t*>(
            malloc(probs.size() * sizeof(whisper_language_prob_t))
        );

        if (!*out_probs) {
            set_error(WHISPER_ERROR_OUT_OF_MEMORY, "Failed to allocate language probabilities");
            return WHISPER_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < probs.size(); ++i) {
            (*out_probs)[i].language = strdup_safe(extract_language_code(probs[i].first));
            (*out_probs)[i].probability = probs[i].second;
        }

        return WHISPER_OK;

    } catch (const std::exception& e) {
        set_error(WHISPER_ERROR_TRANSCRIPTION_FAILED, e.what());
        return WHISPER_ERROR_TRANSCRIPTION_FAILED;
    }
}

void whisper_language_probs_free(whisper_language_prob_t* probs, size_t count) {
    if (probs) {
        for (size_t i = 0; i < count; ++i) {
            free(probs[i].language);
        }
        free(probs);
    }
}

const char* whisper_ct2_version(void) {
    return VERSION;
}

const char* whisper_ct2_ctranslate2_version(void) {
    // CTranslate2 doesn't expose version string at runtime
    // Return minimum supported version (4.x series)
    return "4.x+";
}

const char* whisper_ct2_supported_audio_formats(void) {
    return "wav,flac,ogg,mp3,aiff,au";
}

} // extern "C"
