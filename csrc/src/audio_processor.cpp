#include "audio_processor.h"
#include "stft.h"
#include "mel_filters.h"

#include <sndfile.h>
#include <samplerate.h>

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace whisper_ct2 {

std::vector<float> AudioProcessor::load_audio(const std::string& path) {
    // Open audio file
    SF_INFO sf_info;
    sf_info.format = 0;
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &sf_info);
    if (!sf) {
        throw std::runtime_error("Failed to open audio file: " + path + " - " + sf_strerror(nullptr));
    }

    // Read all samples
    const sf_count_t total_frames = sf_info.frames;
    const int channels = sf_info.channels;
    const int orig_sample_rate = sf_info.samplerate;

    std::vector<float> audio_data(total_frames * channels);
    sf_count_t frames_read = sf_readf_float(sf, audio_data.data(), total_frames);
    sf_close(sf);

    if (frames_read != total_frames) {
        throw std::runtime_error("Failed to read all audio frames");
    }

    // Convert to mono if stereo/multichannel
    std::vector<float> mono_audio(total_frames);
    if (channels == 1) {
        mono_audio = std::move(audio_data);
    } else {
        // Average all channels
        for (sf_count_t i = 0; i < total_frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                sum += audio_data[i * channels + c];
            }
            mono_audio[i] = sum / channels;
        }
    }

    // Resample to 16kHz if needed
    if (orig_sample_rate != SAMPLE_RATE) {
        double ratio = static_cast<double>(SAMPLE_RATE) / orig_sample_rate;
        sf_count_t output_frames = static_cast<sf_count_t>(std::ceil(total_frames * ratio));

        std::vector<float> resampled(output_frames);

        SRC_DATA src_data;
        src_data.data_in = mono_audio.data();
        src_data.input_frames = total_frames;
        src_data.data_out = resampled.data();
        src_data.output_frames = output_frames;
        src_data.src_ratio = ratio;

        // Use high-quality sinc interpolation
        int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
        if (error) {
            throw std::runtime_error("Resampling failed: " + std::string(src_strerror(error)));
        }

        resampled.resize(src_data.output_frames_gen);
        return resampled;
    }

    return mono_audio;
}

std::vector<std::vector<float>> AudioProcessor::compute_mel_spectrogram(
    const std::vector<float>& samples,
    int n_mels
) {
    if (samples.empty()) {
        return std::vector<std::vector<float>>(n_mels);
    }

    // Compute power spectrum via STFT
    auto power_spectrum = STFT::compute_power_spectrum(samples);
    if (power_spectrum.empty() || power_spectrum[0].empty()) {
        return std::vector<std::vector<float>>(n_mels);
    }

    const int n_freqs = static_cast<int>(power_spectrum.size());
    const int n_frames = static_cast<int>(power_spectrum[0].size());

    // Get mel filterbank
    const auto& mel_filters = MelFilters::get(n_mels);

    // Apply mel filterbank: mel = filters @ power_spectrum
    std::vector<std::vector<float>> mel_spec(n_mels, std::vector<float>(n_frames));

    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            float sum = 0.0f;
            for (int f = 0; f < n_freqs && f < static_cast<int>(mel_filters[m].size()); ++f) {
                sum += mel_filters[m][f] * power_spectrum[f][t];
            }
            mel_spec[m][t] = sum;
        }
    }

    // Apply log scale (matches faster-whisper)
    // log_spec = log10(max(mel_spec, 1e-10))
    float max_val = -std::numeric_limits<float>::infinity();
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            mel_spec[m][t] = std::log10(std::max(mel_spec[m][t], 1e-10f));
            max_val = std::max(max_val, mel_spec[m][t]);
        }
    }

    // Dynamic range compression (matches faster-whisper)
    // log_spec = max(log_spec, max_val - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    const float min_val = max_val - 8.0f;
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            mel_spec[m][t] = std::max(mel_spec[m][t], min_val);
            mel_spec[m][t] = (mel_spec[m][t] + 4.0f) / 4.0f;
        }
    }

    return mel_spec;
}

std::vector<std::vector<float>> AudioProcessor::pad_or_trim(
    const std::vector<std::vector<float>>& features,
    int target_frames
) {
    if (features.empty()) {
        return features;
    }

    const int n_mels = static_cast<int>(features.size());
    const int n_frames = static_cast<int>(features[0].size());

    std::vector<std::vector<float>> result(n_mels, std::vector<float>(target_frames, 0.0f));

    const int copy_frames = std::min(n_frames, target_frames);
    for (int m = 0; m < n_mels; ++m) {
        std::copy_n(features[m].begin(), copy_frames, result[m].begin());
    }

    return result;
}

std::vector<float> AudioProcessor::flatten(
    const std::vector<std::vector<float>>& features
) {
    if (features.empty()) {
        return {};
    }

    const int n_mels = static_cast<int>(features.size());
    const int n_frames = static_cast<int>(features[0].size());

    std::vector<float> flat(n_mels * n_frames);
    for (int m = 0; m < n_mels; ++m) {
        std::copy(features[m].begin(), features[m].end(), flat.begin() + m * n_frames);
    }

    return flat;
}

} // namespace whisper_ct2
