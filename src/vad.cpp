#include "vad.hpp"

SimpleVoiceDetector::SimpleVoiceDetector(int sample_rate, int frame_size, int hop_size)
    : sample_rate_(sample_rate), frame_size_(frame_size), hop_size_(hop_size) {}

double SimpleVoiceDetector::calculate_spectral_centroid(const torch::Tensor& audio_frame) {
    auto stft_output = torch::stft(audio_frame, frame_size_, hop_size_, torch::nullopt,
                                   torch::hann_window(frame_size_), false, true, true);
    auto spectrogram = torch::abs(stft_output).pow(2);
    auto freqs = torch::linspace(0, sample_rate_ / 2, spectrogram.size(0));
    auto spectral_centroid = torch::mean((freqs.unsqueeze(1) * spectrogram).sum(0) / (spectrogram.sum(0) + 1e-8));
    return spectral_centroid.item<double>();
}

bool SimpleVoiceDetector::detect_voice(const torch::Tensor& audio_frame) {
    double energy = torch::mean(audio_frame.pow(2)).item<double>();
    double spectral_centroid = calculate_spectral_centroid(audio_frame);
    return (energy > energy_threshold_) && (spectral_centroid > min_speech_freq_) && (spectral_centroid < max_speech_freq_);
}