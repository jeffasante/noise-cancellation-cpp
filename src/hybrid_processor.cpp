#include "hybrid_processor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

HybridNoiseProcessor::HybridNoiseProcessor(const std::string &model_path, int frame_size, int hop_size)
{
    vad_ = std::make_unique<SimpleVoiceDetector>(48000, frame_size, hop_size);
    analyzer_ = std::make_unique<SpectralAnalyzer>(frame_size, hop_size);
    noise_estimator_ = std::make_unique<NoiseEstimator>();
    spectral_subtractor_ = std::make_unique<SpectralSubtractor>();
    ml_processor_ = std::make_unique<MLProcessor>(model_path);

    if (ml_processor_->is_loaded())
    {
        ml_frame_buffer_ = torch::zeros({0}, torch::kFloat32);
        noise_spectrum_ = torch::empty({0});

        // Initialize intelligent processing parameters
        initialize_processing_modes();

        // Set SNR thresholds
        snr_thresholds_["high"] = 15.0;
        snr_thresholds_["medium"] = 5.0;
        snr_thresholds_["low"] = 0.0;

        initialized_ok_ = true;
        std::cout << "Enhanced Hybrid Noise Processor Initialized with Intelligence." << std::endl;
    }
    else
    {
        std::cerr << "Hybrid Processor failed to initialize: ML model could not be loaded." << std::endl;
        initialized_ok_ = false;
    }
}

void HybridNoiseProcessor::initialize_processing_modes()
{
    // Conservative mode: preserve quality, gentle noise reduction
    processing_modes_["conservative"] = {
        "conservative", 0.3, 0.7, 0.5, "Conservative processing for quality preservation"};

    // Balanced mode: equal weight to DSP and ML
    processing_modes_["balanced"] = {
        "balanced", 0.5, 0.5, 1.0, "Balanced processing for general use"};

    // Aggressive mode: strong noise reduction, may affect quality
    processing_modes_["aggressive"] = {
        "aggressive", 0.7, 0.3, 1.5, "Aggressive processing for noisy environments"};
}

bool HybridNoiseProcessor::is_initialized() const
{
    return initialized_ok_;
}

torch::Tensor HybridNoiseProcessor::process_frame(const torch::Tensor &audio_frame)
{
    auto metrics = process_frame_with_metrics(audio_frame);
    // For backward compatibility, return processed audio
    // In practice, you'd extract this from the processing pipeline

    bool is_voice = vad_->detect_voice(audio_frame);
    noise_spectrum_ = noise_estimator_->update_noise_estimate(audio_frame, is_voice, *analyzer_);

    ml_frame_buffer_ = torch::cat({ml_frame_buffer_, audio_frame});
    if (ml_frame_buffer_.size(0) > ml_buffer_target_size_ * 2)
    {
        ml_frame_buffer_ = ml_frame_buffer_.slice(0, ml_frame_buffer_.size(0) - ml_buffer_target_size_);
    }

    frames_since_ml_++;
    if (frames_since_ml_ >= ml_process_interval_ && ml_frame_buffer_.numel() >= ml_buffer_target_size_)
    {
        auto buffer_chunk = ml_frame_buffer_.slice(0, 0, ml_buffer_target_size_);
        auto mask = ml_processor_->process(buffer_chunk);
        if (mask.numel() > 0)
            last_ml_confidence_ = mask.mean().item<double>();
        frames_since_ml_ = 0;
    }

    double adaptive_alpha = 1.0 + (1.0 - last_ml_confidence_) * 4.0;
    adaptive_alpha = std::clamp(adaptive_alpha, 1.0, 3.5);
    spectral_subtractor_->set_alpha(adaptive_alpha);

    return spectral_subtractor_->subtract_noise(audio_frame, noise_spectrum_, *analyzer_);
}

ProcessingMetrics HybridNoiseProcessor::process_frame_with_metrics(const torch::Tensor &audio_frame)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    frame_counter_++;

    // Analyze audio characteristics
    AudioCharacteristics characteristics = analyze_audio_characteristics(audio_frame);

    // Select optimal processing strategy
    ProcessingStrategy strategy = select_processing_strategy(characteristics);

    // Process with DSP
    auto [dsp_output, dsp_metrics] = process_with_dsp(audio_frame, strategy, characteristics);

    // Process with ML (if available)
    auto [ml_output, ml_metrics] = process_with_ml(audio_frame);

    // Intelligent fusion
    auto [final_output, fusion_metrics] = adaptive_fusion(
        audio_frame, dsp_output, ml_output, strategy, dsp_metrics, ml_metrics);

    // Calculate performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double processing_time_ms = duration.count() / 1000.0;

    // Create comprehensive metrics
    ProcessingMetrics metrics = create_metrics(characteristics, strategy, fusion_metrics, processing_time_ms);
    last_metrics_ = metrics;

    return metrics;
}

AudioCharacteristics HybridNoiseProcessor::analyze_audio_characteristics(const torch::Tensor &audio_frame)
{
    AudioCharacteristics characteristics;

    // Voice activity detection
    characteristics.voice_detected = vad_->detect_voice(audio_frame);

    // Get spectral analysis
    auto stft = analyzer_->analyze_spectrum(audio_frame);
    auto magnitude = torch::abs(stft);
    auto power = magnitude.pow(2);
    auto avg_magnitude = magnitude.mean(1);
    auto avg_power = power.mean(1);

    // Estimate SNR
    characteristics.estimated_snr = estimate_snr(audio_frame);

    // Analyze noise type
    auto freq_bins = torch::linspace(0, 24000, magnitude.size(0)); // Assuming 48kHz sample rate
    characteristics.noise_type = analyze_noise_type(avg_magnitude, freq_bins);

    // Calculate spectral complexity
    characteristics.spectral_complexity = calculate_spectral_complexity(avg_power);

    // Determine noise stationarity
    characteristics.noise_stationarity = determine_noise_stationarity(avg_power);

    // Calculate frequency ratios
    int total_bins = avg_power.size(0);
    int low_freq_bins = total_bins / 4;
    int high_freq_bins = total_bins / 2;

    auto low_freq_power = avg_power.slice(0, 0, low_freq_bins).sum();
    auto high_freq_power = avg_power.slice(0, high_freq_bins, total_bins).sum();
    auto total_power = avg_power.sum();

    characteristics.low_freq_ratio = (low_freq_power / total_power).item<double>();
    characteristics.high_freq_ratio = (high_freq_power / total_power).item<double>();

    // Calculate speech energy ratio (simplified)
    // Speech typically in 85Hz - 3400Hz range
    int speech_start_bin = static_cast<int>(85.0 / 24000.0 * total_bins);
    int speech_end_bin = static_cast<int>(3400.0 / 24000.0 * total_bins);
    auto speech_power = avg_power.slice(0, speech_start_bin, speech_end_bin).sum();
    characteristics.speech_energy_ratio = (speech_power / total_power).item<double>();

    return characteristics;
}

std::string HybridNoiseProcessor::analyze_noise_type(const torch::Tensor &spectrum_magnitude, const torch::Tensor &freq_bins)
{
    auto total_power = spectrum_magnitude.sum();

    // Calculate power in different frequency bands
    int total_bins = spectrum_magnitude.size(0);
    int low_band_end = total_bins / 4;    // 0-25% of frequency range
    int high_band_start = total_bins / 2; // 50-100% of frequency range

    auto low_freq_power = spectrum_magnitude.slice(0, 0, low_band_end).sum();
    auto high_freq_power = spectrum_magnitude.slice(0, high_band_start, total_bins).sum();

    double low_freq_ratio = (low_freq_power / total_power).item<double>();
    double high_freq_ratio = (high_freq_power / total_power).item<double>();

    if (low_freq_ratio > 0.6)
    {
        return "low_frequency"; // Traffic, AC, machinery
    }
    else if (high_freq_ratio > 0.4)
    {
        return "broadband"; // White noise, wind
    }
    else
    {
        return "mixed"; // Complex noise environment
    }
}

double HybridNoiseProcessor::calculate_spectral_complexity(const torch::Tensor &spectrum_power)
{
    // Calculate spectral entropy as a measure of complexity
    auto normalized_power = spectrum_power / (spectrum_power.sum() + 1e-8);

    // Avoid log(0) by adding small epsilon
    auto log_power = torch::log(normalized_power + 1e-8);
    auto entropy = -(normalized_power * log_power).sum();

    // Normalize by maximum possible entropy
    double max_entropy = std::log(spectrum_power.size(0));
    double complexity = entropy.item<double>() / max_entropy;

    return std::clamp(complexity, 0.0, 1.0);
}

double HybridNoiseProcessor::estimate_snr(const torch::Tensor &audio_frame)
{
    if (noise_spectrum_.numel() == 0)
    {
        return 10.0; // Default assumption
    }

    // Get current frame spectrum
    auto stft = analyzer_->analyze_spectrum(audio_frame);
    auto magnitude = torch::abs(stft);
    auto signal_power = magnitude.pow(2).mean();

    // Estimate noise power from stored noise spectrum
    auto noise_power = noise_spectrum_.pow(2).mean();

    // Calculate SNR in dB
    double snr_linear = (signal_power / (noise_power + 1e-8)).item<double>();
    return 10.0 * std::log10(snr_linear + 1e-8);
}

std::string HybridNoiseProcessor::determine_noise_stationarity(const torch::Tensor &spectrum_power)
{
    // Simple stationarity check based on power variance
    double power_variance = spectrum_power.var().item<double>();
    return (power_variance < 0.1) ? "stationary" : "non_stationary";
}

ProcessingStrategy HybridNoiseProcessor::select_processing_strategy(const AudioCharacteristics &characteristics)
{
    // Start with base strategy based on SNR
    ProcessingStrategy strategy;

    if (characteristics.estimated_snr > snr_thresholds_["high"])
    {
        strategy = processing_modes_["conservative"];
    }
    else if (characteristics.estimated_snr > snr_thresholds_["medium"])
    {
        strategy = processing_modes_["balanced"];
    }
    else
    {
        strategy = processing_modes_["aggressive"];
    }

    // Adapt strategy based on noise type
    adapt_strategy_for_noise_type(strategy, characteristics.noise_type);

    // Adapt strategy based on voice presence
    adapt_strategy_for_voice(strategy, characteristics.voice_detected);

    // Adapt based on complexity
    if (characteristics.spectral_complexity > 0.8)
    {
        // High complexity: favor ML
        strategy.ml_weight += 0.1;
        strategy.dsp_weight -= 0.1;
    }

    // Normalize weights
    normalize_strategy_weights(strategy);

    // Clamp alpha factor
    strategy.alpha_factor = std::clamp(strategy.alpha_factor, 0.5, 3.0);

    // Update reasoning
    strategy.reasoning = "SNR: " + std::to_string(characteristics.estimated_snr) +
                         "dB, Noise: " + characteristics.noise_type +
                         ", Voice: " + (characteristics.voice_detected ? "true" : "false");

    return strategy;
}

void HybridNoiseProcessor::adapt_strategy_for_noise_type(ProcessingStrategy &strategy, const std::string &noise_type)
{
    if (noise_type == "low_frequency")
    {
        // Low frequency noise: DSP excels
        strategy.dsp_weight += 0.1;
        strategy.ml_weight -= 0.1;
        strategy.alpha_factor *= 1.2;
    }
    else if (noise_type == "broadband")
    {
        // Broadband noise: ML excels
        strategy.ml_weight += 0.1;
        strategy.dsp_weight -= 0.1;
        strategy.alpha_factor *= 0.8;
    }
    // Mixed noise: keep balanced approach
}

void HybridNoiseProcessor::adapt_strategy_for_voice(ProcessingStrategy &strategy, bool voice_detected)
{
    if (voice_detected)
    {
        // Be more conservative when voice is present
        strategy.alpha_factor *= 0.8;
        strategy.ml_weight += 0.1;
        strategy.dsp_weight -= 0.1;
    }
}

void HybridNoiseProcessor::normalize_strategy_weights(ProcessingStrategy &strategy)
{
    double total_weight = strategy.dsp_weight + strategy.ml_weight;
    if (total_weight > 0)
    {
        strategy.dsp_weight /= total_weight;
        strategy.ml_weight /= total_weight;
    }
}

std::pair<torch::Tensor, std::map<std::string, double>> HybridNoiseProcessor::process_with_dsp(
    const torch::Tensor& audio_frame, 
    const ProcessingStrategy& strategy, 
    const AudioCharacteristics& characteristics) {
    
    // Update noise estimate
    noise_spectrum_ = noise_estimator_->update_noise_estimate(audio_frame, characteristics.voice_detected, *analyzer_);
    
    // Set adaptive parameters based on strategy
    double original_alpha = 2.0; // Default alpha from SpectralSubtractor
    spectral_subtractor_->set_alpha(original_alpha * strategy.alpha_factor);
    
    // Apply spectral subtraction
    auto enhanced_audio = spectral_subtractor_->subtract_noise(audio_frame, noise_spectrum_, *analyzer_);
    
    // FIXED: Calculate noise reduction using spectral domain comparison
    std::map<std::string, double> dsp_metrics;
    
    if (noise_spectrum_.numel() > 0) {
        // Method 1: Spectral domain comparison (more accurate)
        auto original_stft = analyzer_->analyze_spectrum(audio_frame);
        auto enhanced_stft = analyzer_->analyze_spectrum(enhanced_audio);
        
        auto original_power = torch::abs(original_stft).pow(2).mean();
        auto enhanced_power = torch::abs(enhanced_stft).pow(2).mean();
        
        // Calculate reduction as power ratio
        double power_ratio = (original_power / (enhanced_power + 1e-8)).item<double>();
        double noise_reduction_db = 10.0 * std::log10(power_ratio + 1e-8);
        
        // Clamp to reasonable range (negative means amplification, which shouldn't happen)
        dsp_metrics["noise_reduction_db"] = std::max(0.0, std::min(noise_reduction_db, 30.0));
        
        // Method 2: Theoretical calculation based on alpha factor (alternative)
        double theoretical_reduction = 20.0 * std::log10(strategy.alpha_factor + 1e-8);
        dsp_metrics["theoretical_reduction_db"] = std::max(0.0, theoretical_reduction);
        
    } else {
        // Fallback when no noise estimate available
        dsp_metrics["noise_reduction_db"] = 0.0;
        dsp_metrics["theoretical_reduction_db"] = 0.0;
    }
    
    dsp_metrics["alpha_used"] = original_alpha * strategy.alpha_factor;
    dsp_metrics["processing_quality"] = 1.0; // Could be enhanced with perceptual metrics
    
    // Restore original alpha
    spectral_subtractor_->set_alpha(original_alpha);
    
    return {enhanced_audio, dsp_metrics};
}


std::pair<torch::Tensor, std::map<std::string, double>> HybridNoiseProcessor::process_with_ml(
    const torch::Tensor &audio_frame)
{

    std::map<std::string, double> ml_metrics;

    if (!ml_processor_->is_loaded())
    {
        ml_metrics["ml_confidence"] = 0.0;
        ml_metrics["method"] = 0.0; // 0 = passthrough
        return {audio_frame, ml_metrics};
    }

    // Update ML buffer
    ml_frame_buffer_ = torch::cat({ml_frame_buffer_, audio_frame});
    if (ml_frame_buffer_.size(0) > ml_buffer_target_size_ * 2)
    {
        ml_frame_buffer_ = ml_frame_buffer_.slice(0, ml_frame_buffer_.size(0) - ml_buffer_target_size_);
    }

    frames_since_ml_++;

    if (frames_since_ml_ >= ml_process_interval_ && ml_frame_buffer_.numel() >= ml_buffer_target_size_)
    {
        auto buffer_chunk = ml_frame_buffer_.slice(0, 0, ml_buffer_target_size_);
        auto mask = ml_processor_->process(buffer_chunk);

        if (mask.numel() > 0)
        {
            last_ml_confidence_ = mask.mean().item<double>();
            double mask_std = mask.std().item<double>();

            ml_metrics["ml_confidence"] = last_ml_confidence_;
            ml_metrics["mask_consistency"] = 1.0 - mask_std; // High consistency = low std
            ml_metrics["method"] = 1.0;                      // 1 = ml_processing
        }

        frames_since_ml_ = 0;
    }
    else
    {
        ml_metrics["ml_confidence"] = last_ml_confidence_;
        ml_metrics["method"] = 2.0; // 2 = cached_confidence
    }

    return {audio_frame, ml_metrics}; // Simplified: return original audio for now
}

std::pair<torch::Tensor, std::map<std::string, double>> HybridNoiseProcessor::adaptive_fusion(
    const torch::Tensor &original_audio,
    const torch::Tensor &dsp_output,
    const torch::Tensor &ml_output,
    const ProcessingStrategy &strategy,
    const std::map<std::string, double> &dsp_metrics,
    const std::map<std::string, double> &ml_metrics)
{

    // Get base weights from strategy
    double dsp_weight = strategy.dsp_weight;
    double ml_weight = strategy.ml_weight;

    // Adjust weights based on ML confidence
    double ml_confidence = ml_metrics.count("ml_confidence") ? ml_metrics.at("ml_confidence") : 0.5;

    if (ml_confidence > quality_threshold_)
    {
        // High ML confidence: trust ML more
        ml_weight *= 1.2;
        dsp_weight *= 0.8;
    }
    else if (ml_confidence < 0.3)
    {
        // Low ML confidence: trust DSP more
        dsp_weight *= 1.2;
        ml_weight *= 0.8;
    }

    // Normalize weights
    double total_weight = dsp_weight + ml_weight;
    if (total_weight > 0)
    {
        dsp_weight /= total_weight;
        ml_weight /= total_weight;
    }

    // Ensure compatible tensor dimensions
    long min_len = std::min({dsp_output.size(0), ml_output.size(0), original_audio.size(0)});
    auto dsp_trimmed = dsp_output.slice(0, 0, min_len);
    auto ml_trimmed = ml_output.slice(0, 0, min_len);

    // Fusion
    auto fused_output = dsp_weight * dsp_trimmed + ml_weight * ml_trimmed;

    // Calculate final metrics
    std::map<std::string, double> fusion_metrics;
    fusion_metrics["final_dsp_weight"] = dsp_weight;
    fusion_metrics["final_ml_weight"] = ml_weight;
    fusion_metrics["ml_confidence"] = ml_confidence;
    fusion_metrics["noise_reduction_db"] = dsp_metrics.count("noise_reduction_db") ? dsp_metrics.at("noise_reduction_db") : 0.0;

    return {fused_output, fusion_metrics};
}

ProcessingMetrics HybridNoiseProcessor::create_metrics(
    const AudioCharacteristics &characteristics,
    const ProcessingStrategy &strategy,
    const std::map<std::string, double> &fusion_metrics,
    double processing_time_ms)
{

    ProcessingMetrics metrics;
    metrics.frame_number = frame_counter_;
    metrics.processing_time_ms = processing_time_ms;
    metrics.estimated_snr = characteristics.estimated_snr;
    metrics.noise_type = characteristics.noise_type;
    metrics.strategy_used = strategy.mode;
    metrics.reasoning = strategy.reasoning;

    // Extract fusion metrics
    if (fusion_metrics.count("final_dsp_weight"))
    {
        metrics.final_dsp_weight = fusion_metrics.at("final_dsp_weight");
    }
    if (fusion_metrics.count("final_ml_weight"))
    {
        metrics.final_ml_weight = fusion_metrics.at("final_ml_weight");
    }
    if (fusion_metrics.count("ml_confidence"))
    {
        metrics.ml_confidence = fusion_metrics.at("ml_confidence");
    }
    if (fusion_metrics.count("noise_reduction_db"))
    {
        metrics.noise_reduction_db = fusion_metrics.at("noise_reduction_db");
    }

    return metrics;
}

void HybridNoiseProcessor::set_processing_mode(const std::string &mode)
{
    if (processing_modes_.count(mode))
    {
        // Update default thresholds based on mode
        if (mode == "conservative")
        {
            quality_threshold_ = 0.9;
        }
        else if (mode == "aggressive")
        {
            quality_threshold_ = 0.6;
        }
        else
        {
            quality_threshold_ = 0.8; // balanced
        }
    }
}