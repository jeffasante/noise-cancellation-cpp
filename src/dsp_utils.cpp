#include "dsp_utils.hpp"
#include <algorithm>

SpectralAnalyzer::SpectralAnalyzer(int n_fft, int hop_length) : n_fft_(n_fft), hop_length_(hop_length) {}

torch::Tensor SpectralAnalyzer::analyze_spectrum(const torch::Tensor& audio) {
    long pad_amount = n_fft_ / 2;
    auto audio_3d = audio.unsqueeze(0).unsqueeze(0); 
    auto padded_audio_3d = torch::nn::functional::pad(audio_3d, torch::nn::functional::PadFuncOptions({pad_amount, pad_amount}).mode(torch::kReflect));
    auto padded_audio = padded_audio_3d.squeeze(0).squeeze(0);
    return torch::stft(padded_audio, n_fft_, hop_length_, n_fft_, torch::hann_window(n_fft_), false, true, true);
}

torch::Tensor SpectralAnalyzer::reconstruct_audio(const torch::Tensor& stft, int original_length) {
    return torch::istft(stft, n_fft_, hop_length_, n_fft_, torch::hann_window(n_fft_), true, false, true, original_length);
}

std::pair<torch::Tensor, torch::Tensor> SpectralAnalyzer::get_magnitude_and_phase(const torch::Tensor& audio) {
    auto stft = analyze_spectrum(audio);
    auto magnitude = torch::abs(stft);
    auto phase = torch::angle(stft);
    return {magnitude, phase};
}

torch::Tensor SpectralAnalyzer::get_power_spectrum(const torch::Tensor& audio) {
    auto [magnitude, phase] = get_magnitude_and_phase(audio);
    return magnitude.pow(2);
}

torch::Tensor SpectralAnalyzer::get_frequency_bins(int sample_rate) {
    return torch::linspace(0, sample_rate / 2.0, n_fft_ / 2 + 1);
}

NoiseEstimator::NoiseEstimator() {}

torch::Tensor NoiseEstimator::update_noise_estimate(const torch::Tensor& frame, bool is_voice, SpectralAnalyzer& analyzer) {
    auto stft = analyzer.analyze_spectrum(frame);
    auto current_mag = torch::abs(stft).mean(1);
    
    if (!is_initialized_) {
        noise_spectrum_ = current_mag.clone();
        is_initialized_ = true;
        frames_processed_ = 1;
    } else if (!is_voice) {
        // Adaptive update with configurable rate
        noise_spectrum_ = (1.0 - noise_update_rate_) * noise_spectrum_ + noise_update_rate_ * current_mag;
        frames_processed_++;
    }
    
    return noise_spectrum_;
}

bool NoiseEstimator::is_estimate_reliable() const {
    return is_initialized_ && frames_processed_ >= min_frames_for_reliability_;
}

torch::Tensor NoiseEstimator::get_noise_estimate() const {
    return noise_spectrum_;
}

void NoiseEstimator::reset_estimate() {
    is_initialized_ = false;
    frames_processed_ = 0;
    noise_spectrum_ = torch::empty({0});
}

SpectralSubtractor::SpectralSubtractor() {}

void SpectralSubtractor::set_alpha(double new_alpha) {
    alpha_ = new_alpha;
}

torch::Tensor SpectralSubtractor::subtract_noise(const torch::Tensor& audio, const torch::Tensor& noise_spec, SpectralAnalyzer& analyzer) {
    if (noise_spec.numel() == 0) return audio;
    
    auto stft = analyzer.analyze_spectrum(audio);
    auto mag = torch::abs(stft);
    auto phase = torch::angle(stft);
    auto noise_mag = noise_spec.unsqueeze(1).expand_as(mag);
    
    torch::Tensor alpha_tensor;
    
    if (adaptive_alpha_enabled_) {
        // Calculate SNR for adaptive alpha
        auto snr = mag / (noise_mag + 1e-8);
        alpha_tensor = calculate_adaptive_alpha(snr);
        
        if (freq_dependent_enabled_) {
            alpha_tensor = apply_frequency_dependent_processing(alpha_tensor);
        }
    } else {
        // Use fixed alpha
        alpha_tensor = torch::full_like(mag, alpha_);
    }
    
    // Enhanced spectral subtraction with adaptive parameters
    auto enhanced_mag = torch::max(mag - alpha_tensor * noise_mag, beta_ * mag);
    auto enhanced_stft = torch::polar(enhanced_mag, phase);
    
    return analyzer.reconstruct_audio(enhanced_stft, audio.size(0));
}

torch::Tensor SpectralSubtractor::calculate_adaptive_alpha(const torch::Tensor& snr) {
    // Adaptive alpha based on SNR
    // High SNR: conservative subtraction (lower alpha)
    // Low SNR: aggressive subtraction (higher alpha)
    auto alpha_adaptive = torch::where(
        snr > 10,  // High SNR
        torch::full_like(snr, 1.0),  // Conservative
        torch::where(
            snr > 3,  // Medium SNR
            torch::full_like(snr, alpha_),  // Normal
            torch::full_like(snr, alpha_ * 1.5)  // Aggressive
        )
    );
    
    return alpha_adaptive;
}

torch::Tensor SpectralSubtractor::apply_frequency_dependent_processing(const torch::Tensor& alpha_map) {
    // More aggressive processing in low frequencies (where noise is often stronger)
    auto freq_weights = torch::ones_like(alpha_map);
    
    // Boost low frequencies (first 25% of spectrum)
    int low_freq_bins = alpha_map.size(0) / 4;
    freq_weights.slice(0, 0, low_freq_bins) *= 1.3;
    
    // Reduce high frequencies (last 25% of spectrum) to preserve speech clarity
    int high_freq_start = alpha_map.size(0) * 3 / 4;
    freq_weights.slice(0, high_freq_start, alpha_map.size(0)) *= 0.8;
    
    return alpha_map * freq_weights;
}