#ifndef DSP_UTILS_HPP
#define DSP_UTILS_HPP
#include <torch/torch.h>

class SpectralAnalyzer {
public:
    SpectralAnalyzer(int n_fft, int hop_length);
    torch::Tensor analyze_spectrum(const torch::Tensor& audio);
    torch::Tensor reconstruct_audio(const torch::Tensor& stft, int original_length);
    
    // Enhanced analysis methods
    std::pair<torch::Tensor, torch::Tensor> get_magnitude_and_phase(const torch::Tensor& audio);
    torch::Tensor get_power_spectrum(const torch::Tensor& audio);
    torch::Tensor get_frequency_bins(int sample_rate = 48000);
    
private:
    int n_fft_;
    int hop_length_;
};

class NoiseEstimator {
public:
    NoiseEstimator();
    torch::Tensor update_noise_estimate(const torch::Tensor& frame, bool is_voice, SpectralAnalyzer& analyzer);
    
    // Enhanced noise estimation
    bool is_estimate_reliable() const;
    torch::Tensor get_noise_estimate() const;
    void reset_estimate();
    double get_noise_update_rate() const { return noise_update_rate_; }
    void set_noise_update_rate(double rate) { noise_update_rate_ = rate; }
    
private:
    torch::Tensor noise_spectrum_;
    bool is_initialized_ = false;
    int frames_processed_ = 0;
    double noise_update_rate_ = 0.02;
    static constexpr int min_frames_for_reliability_ = 10;
};

class SpectralSubtractor {
public:
    SpectralSubtractor();
    torch::Tensor subtract_noise(const torch::Tensor& audio, const torch::Tensor& noise_spec, SpectralAnalyzer& analyzer);
    void set_alpha(double new_alpha);
    
    // Enhanced spectral subtraction
    void set_beta(double new_beta) { beta_ = new_beta; }
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    
    // Advanced processing options
    void enable_adaptive_alpha(bool enable) { adaptive_alpha_enabled_ = enable; }
    void set_frequency_dependent_processing(bool enable) { freq_dependent_enabled_ = enable; }
    
private:
    double alpha_ = 2.0;
    double beta_ = 0.01;
    bool adaptive_alpha_enabled_ = true;
    bool freq_dependent_enabled_ = true;
    
    // Helper methods for enhanced processing
    torch::Tensor calculate_adaptive_alpha(const torch::Tensor& snr);
    torch::Tensor apply_frequency_dependent_processing(const torch::Tensor& alpha_map);
};

#endif