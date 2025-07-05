#ifndef HYBRID_PROCESSOR_HPP
#define HYBRID_PROCESSOR_HPP
#include "vad.hpp"
#include "dsp_utils.hpp"
#include "ml_processor.hpp"
#include <memory>
#include <string>
#include <map>
#include <chrono>

// Structures for intelligent processing
struct AudioCharacteristics {
    bool voice_detected = false;
    double estimated_snr = 10.0;
    std::string noise_type = "mixed";
    std::string noise_stationarity = "stationary";
    double speech_energy_ratio = 0.5;
    double spectral_complexity = 0.5;
    double low_freq_ratio = 0.3;
    double high_freq_ratio = 0.3;
};

struct ProcessingStrategy {
    std::string mode = "balanced";
    double dsp_weight = 0.5;
    double ml_weight = 0.5;
    double alpha_factor = 1.0;
    std::string reasoning = "";
};

struct ProcessingMetrics {
    double processing_time_ms = 0.0;
    double estimated_snr = 0.0;
    std::string noise_type = "";
    std::string strategy_used = "";
    double final_dsp_weight = 0.0;
    double final_ml_weight = 0.0;
    double ml_confidence = 0.0;
    double noise_reduction_db = 0.0;
    std::string reasoning = "";
    long frame_number = 0;
};

class HybridNoiseProcessor {
public:
    HybridNoiseProcessor(const std::string& model_path, int frame_size, int hop_size);
    torch::Tensor process_frame(const torch::Tensor& audio_frame);
    bool is_initialized() const;
    
    // New intelligent processing methods
    ProcessingMetrics process_frame_with_metrics(const torch::Tensor& audio_frame);
    AudioCharacteristics analyze_audio_characteristics(const torch::Tensor& audio_frame);
    ProcessingStrategy select_processing_strategy(const AudioCharacteristics& characteristics);
    
    // Configuration methods
    void set_quality_threshold(double threshold) { quality_threshold_ = threshold; }
    void set_processing_mode(const std::string& mode);
    ProcessingMetrics get_last_metrics() const { return last_metrics_; }

private:
    // Core components
    std::unique_ptr<SimpleVoiceDetector> vad_;
    std::unique_ptr<NoiseEstimator> noise_estimator_;
    std::unique_ptr<SpectralSubtractor> spectral_subtractor_;
    std::unique_ptr<MLProcessor> ml_processor_;
    std::unique_ptr<SpectralAnalyzer> analyzer_;
    
    // Processing state
    torch::Tensor ml_frame_buffer_;
    long ml_buffer_target_size_ = 96000;
    int frames_since_ml_ = 0;
    const int ml_process_interval_ = 20;
    double last_ml_confidence_ = 0.8;
    torch::Tensor noise_spectrum_;
    bool initialized_ok_ = false;
    
    // New intelligent processing parameters
    double quality_threshold_ = 0.8;
    std::map<std::string, ProcessingStrategy> processing_modes_;
    std::map<std::string, double> snr_thresholds_;
    ProcessingMetrics last_metrics_;
    long frame_counter_ = 0;
    
    // Analysis and strategy methods
    std::string analyze_noise_type(const torch::Tensor& spectrum_magnitude, const torch::Tensor& freq_bins);
    double calculate_spectral_complexity(const torch::Tensor& spectrum_power);
    double estimate_snr(const torch::Tensor& audio_frame);
    std::string determine_noise_stationarity(const torch::Tensor& spectrum_power);
    
    // Processing methods
    std::pair<torch::Tensor, std::map<std::string, double>> process_with_dsp(
        const torch::Tensor& audio_frame, 
        const ProcessingStrategy& strategy, 
        const AudioCharacteristics& characteristics
    );
    
    std::pair<torch::Tensor, std::map<std::string, double>> process_with_ml(
        const torch::Tensor& audio_frame
    );
    
    std::pair<torch::Tensor, std::map<std::string, double>> adaptive_fusion(
        const torch::Tensor& original_audio,
        const torch::Tensor& dsp_output,
        const torch::Tensor& ml_output,
        const ProcessingStrategy& strategy,
        const std::map<std::string, double>& dsp_metrics,
        const std::map<std::string, double>& ml_metrics
    );
    
    // Utility methods
    void initialize_processing_modes();
    void adapt_strategy_for_noise_type(ProcessingStrategy& strategy, const std::string& noise_type);
    void adapt_strategy_for_voice(ProcessingStrategy& strategy, bool voice_detected);
    void normalize_strategy_weights(ProcessingStrategy& strategy);
    ProcessingMetrics create_metrics(
        const AudioCharacteristics& characteristics,
        const ProcessingStrategy& strategy,
        const std::map<std::string, double>& fusion_metrics,
        double processing_time_ms
    );
};

#endif