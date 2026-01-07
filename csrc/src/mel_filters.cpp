#include "mel_filters.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace whisper_ct2 {

// Static member initialization
std::vector<std::vector<float>> MelFilters::mel_80_;
std::vector<std::vector<float>> MelFilters::mel_128_;
bool MelFilters::initialized_80_ = false;
bool MelFilters::initialized_128_ = false;

float MelFilters::hz_to_mel(float hz) {
    // Slaney formula (used by librosa and faster-whisper)
    // Below 1000 Hz: linear
    // Above 1000 Hz: logarithmic
    constexpr float f_min = 0.0f;
    constexpr float f_sp = 200.0f / 3.0f;  // ~66.67 Hz
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = (min_log_hz - f_min) / f_sp;
    constexpr float logstep = std::log(6.4f) / 27.0f;  // log(6400/1000) / 27

    if (hz >= min_log_hz) {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return (hz - f_min) / f_sp;
}

float MelFilters::mel_to_hz(float mel) {
    // Inverse of hz_to_mel (Slaney formula)
    constexpr float f_min = 0.0f;
    constexpr float f_sp = 200.0f / 3.0f;
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = (min_log_hz - f_min) / f_sp;
    constexpr float logstep = std::log(6.4f) / 27.0f;

    if (mel >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return f_min + f_sp * mel;
}

std::vector<std::vector<float>> MelFilters::compute_mel_filterbank(int n_mels) {
    // Matching librosa.filters.mel with htk=False (Slaney formula)
    constexpr float fmin = 0.0f;
    constexpr float fmax = 8000.0f;  // sr / 2
    constexpr int n_fft = N_FFT;
    constexpr int sr = SAMPLE_RATE;

    // Number of frequency bins
    const int n_freqs = n_fft / 2 + 1;

    // Frequency resolution
    const float fft_freqs_step = static_cast<float>(sr) / n_fft;

    // FFT bin frequencies
    std::vector<float> fft_freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        fft_freqs[i] = i * fft_freqs_step;
    }

    // Mel points (n_mels + 2 points for triangular filters)
    const float mel_min = hz_to_mel(fmin);
    const float mel_max = hz_to_mel(fmax);
    const int n_mels_plus_2 = n_mels + 2;

    std::vector<float> mel_points(n_mels_plus_2);
    for (int i = 0; i < n_mels_plus_2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels_plus_2 - 1);
    }

    // Convert mel points back to Hz
    std::vector<float> hz_points(n_mels_plus_2);
    for (int i = 0; i < n_mels_plus_2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Create filterbank
    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(n_freqs, 0.0f));

    for (int m = 0; m < n_mels; ++m) {
        const float f_left = hz_points[m];
        const float f_center = hz_points[m + 1];
        const float f_right = hz_points[m + 2];

        for (int k = 0; k < n_freqs; ++k) {
            const float freq = fft_freqs[k];

            if (freq >= f_left && freq <= f_center) {
                // Rising slope
                if (f_center != f_left) {
                    filterbank[m][k] = (freq - f_left) / (f_center - f_left);
                }
            } else if (freq > f_center && freq <= f_right) {
                // Falling slope
                if (f_right != f_center) {
                    filterbank[m][k] = (f_right - freq) / (f_right - f_center);
                }
            }
        }

        // Normalize by filter width (Slaney normalization)
        // This matches librosa's norm='slaney' behavior
        const float enorm = 2.0f / (hz_points[m + 2] - hz_points[m]);
        for (int k = 0; k < n_freqs; ++k) {
            filterbank[m][k] *= enorm;
        }
    }

    return filterbank;
}

const std::vector<std::vector<float>>& MelFilters::get(int n_mels) {
    if (n_mels == 80) {
        if (!initialized_80_) {
            mel_80_ = compute_mel_filterbank(80);
            initialized_80_ = true;
        }
        return mel_80_;
    } else if (n_mels == 128) {
        if (!initialized_128_) {
            mel_128_ = compute_mel_filterbank(128);
            initialized_128_ = true;
        }
        return mel_128_;
    } else {
        throw std::invalid_argument("n_mels must be 80 or 128");
    }
}

} // namespace whisper_ct2
