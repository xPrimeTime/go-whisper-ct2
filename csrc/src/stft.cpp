#include "stft.h"
#include "pocketfft_hdronly.h"
#include <cmath>
#include <algorithm>

namespace whisper_ct2 {

// Static member initialization
std::vector<float> STFT::hann_window_;
bool STFT::hann_initialized_ = false;

const std::vector<float>& STFT::hann_window() {
    if (!hann_initialized_) {
        // Compute periodic Hann window matching numpy/librosa
        // window[i] = 0.5 * (1 - cos(2*pi*i/N))
        hann_window_.resize(N_FFT);
        const double scale = 2.0 * M_PI / N_FFT;
        for (int i = 0; i < N_FFT; ++i) {
            hann_window_[i] = static_cast<float>(0.5 * (1.0 - std::cos(scale * i)));
        }
        hann_initialized_ = true;
    }
    return hann_window_;
}

int STFT::num_frames(int audio_length) {
    // Match librosa's frame calculation
    // With center=True (default), audio is padded by n_fft//2 on each side
    // n_frames = 1 + (padded_length - n_fft) // hop_length
    // padded_length = audio_length + n_fft
    // n_frames = 1 + audio_length // hop_length
    return 1 + audio_length / HOP_LENGTH;
}

std::vector<std::vector<float>> STFT::compute_power_spectrum(
    const std::vector<float>& audio
) {
    if (audio.empty()) {
        return {};
    }

    const auto& window = hann_window();
    const int n_frames = num_frames(static_cast<int>(audio.size()));
    const int pad_amount = N_FFT / 2;

    // Create padded audio (reflect padding like librosa center=True)
    std::vector<float> padded(audio.size() + N_FFT);

    // Reflect padding at start
    for (int i = 0; i < pad_amount; ++i) {
        int src_idx = pad_amount - i;
        if (src_idx >= static_cast<int>(audio.size())) {
            src_idx = static_cast<int>(audio.size()) - 1;
        }
        padded[i] = audio[src_idx];
    }

    // Copy original audio
    std::copy(audio.begin(), audio.end(), padded.begin() + pad_amount);

    // Reflect padding at end
    for (int i = 0; i < pad_amount; ++i) {
        int src_idx = static_cast<int>(audio.size()) - 2 - i;
        if (src_idx < 0) {
            src_idx = 0;
        }
        padded[pad_amount + audio.size() + i] = audio[src_idx];
    }

    // Output: [n_freqs][n_frames]
    std::vector<std::vector<float>> power_spectrum(N_FREQS, std::vector<float>(n_frames));

    // Prepare FFT
    pocketfft::shape_t shape = {static_cast<size_t>(N_FFT)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};
    pocketfft::shape_t axes = {0};

    // Buffers for FFT
    std::vector<float> frame(N_FFT);
    std::vector<std::complex<float>> fft_out(N_FFT / 2 + 1);

    // Process each frame
    for (int t = 0; t < n_frames; ++t) {
        const int start = t * HOP_LENGTH;

        // Extract and window the frame
        for (int i = 0; i < N_FFT; ++i) {
            int idx = start + i;
            if (idx < static_cast<int>(padded.size())) {
                frame[i] = padded[idx] * window[i];
            } else {
                frame[i] = 0.0f;
            }
        }

        // Compute real FFT
        pocketfft::r2c(
            shape,
            stride_in,
            stride_out,
            axes,
            pocketfft::FORWARD,
            frame.data(),
            fft_out.data(),
            1.0f
        );

        // Compute power spectrum (magnitude squared)
        for (int f = 0; f < N_FREQS; ++f) {
            float re = fft_out[f].real();
            float im = fft_out[f].imag();
            power_spectrum[f][t] = re * re + im * im;
        }
    }

    return power_spectrum;
}

} // namespace whisper_ct2
