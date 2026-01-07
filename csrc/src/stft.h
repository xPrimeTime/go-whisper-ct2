#ifndef WHISPER_CT2_STFT_H
#define WHISPER_CT2_STFT_H

#include <vector>
#include <complex>
#include <cstdint>

namespace whisper_ct2 {

// Short-Time Fourier Transform implementation
// Matches librosa/faster-whisper STFT behavior
class STFT {
public:
    // Whisper STFT parameters
    static constexpr int N_FFT = 400;
    static constexpr int HOP_LENGTH = 160;
    static constexpr int N_FREQS = N_FFT / 2 + 1;  // 201

    // Compute STFT magnitude squared (power spectrum)
    // Input: audio samples (float32, mono, any length)
    // Output: [n_freqs][n_frames] power spectrum
    static std::vector<std::vector<float>> compute_power_spectrum(
        const std::vector<float>& audio
    );

    // Compute number of output frames for given input length
    static int num_frames(int audio_length);

private:
    // Get or compute Hann window
    static const std::vector<float>& hann_window();

    // Cached Hann window
    static std::vector<float> hann_window_;
    static bool hann_initialized_;
};

} // namespace whisper_ct2

#endif // WHISPER_CT2_STFT_H
