#ifndef WHISPER_CT2_AUDIO_PROCESSOR_H
#define WHISPER_CT2_AUDIO_PROCESSOR_H

#include <vector>
#include <string>
#include <cstdint>

namespace whisper_ct2 {

// Audio processing pipeline matching faster-whisper
class AudioProcessor {
public:
    // Whisper audio parameters
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int N_FFT = 400;
    static constexpr int HOP_LENGTH = 160;
    static constexpr int CHUNK_LENGTH = 30;  // seconds
    static constexpr int N_SAMPLES_PER_CHUNK = CHUNK_LENGTH * SAMPLE_RATE;  // 480000
    static constexpr int N_FRAMES_PER_CHUNK = N_SAMPLES_PER_CHUNK / HOP_LENGTH;  // 3000

    // Load audio file, resample to 16kHz mono
    // Supported formats: WAV, FLAC, OGG, MP3, etc. (via libsndfile)
    // Returns: PCM float32 samples normalized to [-1, 1]
    // Throws: std::runtime_error on failure
    static std::vector<float> load_audio(const std::string& path);

    // Compute log-mel spectrogram (matches faster-whisper exactly)
    // Input: mono 16kHz float32 samples
    // Output: [n_mels][n_frames] mel spectrogram
    static std::vector<std::vector<float>> compute_mel_spectrogram(
        const std::vector<float>& samples,
        int n_mels = 80
    );

    // Pad or trim mel spectrogram to standard chunk length
    // Input: [n_mels][any_frames]
    // Output: [n_mels][N_FRAMES_PER_CHUNK]
    static std::vector<std::vector<float>> pad_or_trim(
        const std::vector<std::vector<float>>& features,
        int target_frames = N_FRAMES_PER_CHUNK
    );

    // Convert mel spectrogram to flat vector for CTranslate2
    // Input: [n_mels][n_frames]
    // Output: flat vector [n_mels * n_frames]
    static std::vector<float> flatten(
        const std::vector<std::vector<float>>& features
    );

    // Get audio duration in seconds from sample count
    static float get_duration(size_t sample_count) {
        return static_cast<float>(sample_count) / SAMPLE_RATE;
    }
};

} // namespace whisper_ct2

#endif // WHISPER_CT2_AUDIO_PROCESSOR_H
