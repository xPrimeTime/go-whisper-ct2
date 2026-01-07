#ifndef WHISPER_CT2_MEL_FILTERS_H
#define WHISPER_CT2_MEL_FILTERS_H

#include <vector>
#include <cstdint>

namespace whisper_ct2 {

// Mel filterbank computation matching librosa/faster-whisper
// Parameters:
//   sr = 16000 (sample rate)
//   n_fft = 400 (FFT size)
//   n_mels = 80 or 128 (number of mel bands)
//   fmin = 0 (minimum frequency)
//   fmax = 8000 (maximum frequency = sr/2)

class MelFilters {
public:
    // Get mel filterbank for given number of mel bins
    // Returns [n_mels][n_freqs] where n_freqs = n_fft/2 + 1 = 201
    static const std::vector<std::vector<float>>& get(int n_mels);

    // Number of frequency bins (n_fft/2 + 1)
    static constexpr int N_FREQS = 201;

    // Sample rate
    static constexpr int SAMPLE_RATE = 16000;

    // FFT size
    static constexpr int N_FFT = 400;

private:
    // Compute mel filterbank (called once, cached)
    static std::vector<std::vector<float>> compute_mel_filterbank(int n_mels);

    // Convert frequency to mel scale
    static float hz_to_mel(float hz);

    // Convert mel to frequency
    static float mel_to_hz(float mel);

    // Cached filterbanks
    static std::vector<std::vector<float>> mel_80_;
    static std::vector<std::vector<float>> mel_128_;
    static bool initialized_80_;
    static bool initialized_128_;
};

} // namespace whisper_ct2

#endif // WHISPER_CT2_MEL_FILTERS_H
