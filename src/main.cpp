#include "hybrid_processor.hpp" 
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <sndfile.h> // For loading and saving audio files
#include <map>
#include <iomanip>
#include <fstream>

// Audio I/O Helper Functions
/**
 * @brief Loads an audio file from the given path into a mono torch::Tensor.
 *
 * @param path The file path to the audio file.
 * @param sample_rate A reference to an integer that will be populated with the audio's sample rate.
 * @return A 1D torch::Tensor containing the audio data.
 */
torch::Tensor load_audio(const std::string &path, int &sample_rate)
{
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!infile)
    {
        std::cerr << "Error: Could not open input file: " << path << std::endl;
        return torch::empty({0});
    }

    std::vector<float> buffer(sfinfo.frames * sfinfo.channels);
    sf_read_float(infile, buffer.data(), buffer.size());
    sf_close(infile);

    sample_rate = sfinfo.samplerate;
    torch::Tensor tensor = torch::from_blob(buffer.data(), {sfinfo.frames, sfinfo.channels}).clone();

    // If the audio has multiple channels (e.g., stereo), convert to mono by averaging them.
    if (tensor.size(1) > 1)
    {
        tensor = tensor.mean(1);
    }

    std::cout << "Loaded audio file: " << path << " (" << sfinfo.frames << " samples, " << sfinfo.samplerate << " Hz)" << std::endl;

    // Squeeze to ensure the final tensor is 1-dimensional.
    return tensor.squeeze();
}

/**
 * @brief Saves a 1D torch::Tensor to a WAV file.
 *
 * @param path The path where the output file will be saved.
 * @param tensor The 1D tensor containing the audio data to save.
 * @param sample_rate The sample rate of the audio.
 */
void save_audio(const std::string &path, const torch::Tensor &tensor, int sample_rate)
{
    SF_INFO sfinfo;
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = 1;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16; // Standard 16-bit WAV format

    SNDFILE *outfile = sf_open(path.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile)
    {
        std::cerr << "Error: Could not open output file: " << path << std::endl;
        return;
    }

    // Ensure the tensor's memory is contiguous before getting the data pointer.
    auto contiguous_tensor = tensor.contiguous();
    std::vector<float> buffer(contiguous_tensor.data_ptr<float>(), contiguous_tensor.data_ptr<float>() + contiguous_tensor.numel());

    sf_write_float(outfile, buffer.data(), buffer.size());
    sf_close(outfile);

    std::cout << "\nEnhanced audio saved to: " << path << std::endl;
}

// Enhanced Metrics and Analysis Functions
/**
 * @brief Calculate summary statistics from processing metrics
 */
struct SummaryMetrics
{
    double avg_processing_time_ms = 0.0;
    double avg_estimated_snr = 0.0;
    double avg_dsp_weight = 0.0;
    double avg_ml_weight = 0.0;
    double avg_noise_reduction_db = 0.0;
    std::map<std::string, int> strategy_distribution;
    std::map<std::string, int> noise_type_distribution;
    int total_frames = 0;
};

SummaryMetrics calculate_summary_metrics(const std::vector<ProcessingMetrics> &all_metrics)
{
    SummaryMetrics summary;

    if (all_metrics.empty())
    {
        return summary;
    }

    // Calculate averages
    double total_processing_time = 0.0;
    double total_snr = 0.0;
    double total_dsp_weight = 0.0;
    double total_ml_weight = 0.0;
    double total_noise_reduction = 0.0;

    for (const auto &metric : all_metrics)
    {
        total_processing_time += metric.processing_time_ms;
        total_snr += metric.estimated_snr;
        total_dsp_weight += metric.final_dsp_weight;
        total_ml_weight += metric.final_ml_weight;
        total_noise_reduction += metric.noise_reduction_db;

        // Count distributions
        summary.strategy_distribution[metric.strategy_used]++;
        summary.noise_type_distribution[metric.noise_type]++;
    }

    summary.total_frames = all_metrics.size();
    summary.avg_processing_time_ms = total_processing_time / summary.total_frames;
    summary.avg_estimated_snr = total_snr / summary.total_frames;
    summary.avg_dsp_weight = total_dsp_weight / summary.total_frames;
    summary.avg_ml_weight = total_ml_weight / summary.total_frames;
    summary.avg_noise_reduction_db = total_noise_reduction / summary.total_frames;

    return summary;
}

/**
 * @brief Print detailed metrics analysis
 */
void print_detailed_analysis(const SummaryMetrics &summary)
{
    std::cout << "\n"
              << std::string(60, '=') << std::endl;
    std::cout << "INTELLIGENT PROCESSING ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "  Average processing time: " << summary.avg_processing_time_ms << " ms per frame" << std::endl;
    std::cout << "  Average estimated SNR: " << summary.avg_estimated_snr << " dB" << std::endl;
    std::cout << "  Average noise reduction: " << summary.avg_noise_reduction_db << " dB" << std::endl;
    std::cout << "  Total frames processed: " << summary.total_frames << std::endl;

    std::cout << "\nIntelligent Fusion Analysis:" << std::endl;
    std::cout << "  Average DSP weight: " << summary.avg_dsp_weight << std::endl;
    std::cout << "  Average ML weight: " << summary.avg_ml_weight << std::endl;

    std::cout << "\nStrategy Distribution:" << std::endl;
    for (const auto &[strategy, count] : summary.strategy_distribution)
    {
        double percentage = (count * 100.0) / summary.total_frames;
        std::cout << "  " << strategy << ": " << count << " frames (" << percentage << "%)" << std::endl;
    }

    std::cout << "\nNoise Type Distribution:" << std::endl;
    for (const auto &[noise_type, count] : summary.noise_type_distribution)
    {
        double percentage = (count * 100.0) / summary.total_frames;
        std::cout << "  " << noise_type << ": " << count << " frames (" << percentage << "%)" << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;
}

/**
 * @brief Save metrics to JSON file for further analysis
 */
void save_metrics_to_file(const std::vector<ProcessingMetrics> &all_metrics,
                          const SummaryMetrics &summary,
                          const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not save metrics to " << filename << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"summary\": {\n";
    file << "    \"avg_processing_time_ms\": " << summary.avg_processing_time_ms << ",\n";
    file << "    \"avg_estimated_snr\": " << summary.avg_estimated_snr << ",\n";
    file << "    \"avg_dsp_weight\": " << summary.avg_dsp_weight << ",\n";
    file << "    \"avg_ml_weight\": " << summary.avg_ml_weight << ",\n";
    file << "    \"avg_noise_reduction_db\": " << summary.avg_noise_reduction_db << ",\n";
    file << "    \"total_frames\": " << summary.total_frames << "\n";
    file << "  },\n";

    file << "  \"strategy_distribution\": {\n";
    bool first = true;
    for (const auto &[strategy, count] : summary.strategy_distribution)
    {
        if (!first)
            file << ",\n";
        file << "    \"" << strategy << "\": " << count;
        first = false;
    }
    file << "\n  },\n";

    file << "  \"noise_type_distribution\": {\n";
    first = true;
    for (const auto &[noise_type, count] : summary.noise_type_distribution)
    {
        if (!first)
            file << ",\n";
        file << "    \"" << noise_type << "\": " << count;
        first = false;
    }
    file << "\n  }\n";
    file << "}\n";

    file.close();
    std::cout << "Detailed metrics saved to: " << filename << std::endl;
}

// Main Application Logic
int main(int argc, char *argv[])
{
    // Check for the correct number of command-line arguments
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file.wav> <output_file.wav> [processing_mode]" << std::endl;
        std::cerr << "Processing modes: conservative, balanced, aggressive (default: balanced)" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string processing_mode = (argc > 3) ? argv[3] : "balanced";
    std::string model_path = "traced_denoiser_model.pt"; // Assumes model is in the run directory

    // Validate processing mode
    if (processing_mode != "conservative" && processing_mode != "balanced" && processing_mode != "aggressive")
    {
        std::cerr << "Warning: Invalid processing mode '" << processing_mode << "'. Using 'balanced'." << std::endl;
        processing_mode = "balanced";
    }

    std::cout << "Enhanced Hybrid Noise Cancellation System" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Input file: " << input_path << std::endl;
    std::cout << "Output file: " << output_path << std::endl;
    std::cout << "Processing mode: " << processing_mode << std::endl;
    std::cout << "ML model: " << model_path << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load Audio
    int sample_rate;
    torch::Tensor audio_data = load_audio(input_path, sample_rate);
    if (audio_data.numel() == 0)
    {
        return 1; // Exit if loading failed
    }

    // Initialize Enhanced Processor
    long frame_size = 1024;
    long hop_size = 512;

    HybridNoiseProcessor processor(model_path, frame_size, hop_size);

    // This is the critical check to prevent crashes from initialization failures.
    if (!processor.is_initialized())
    {
        std::cerr << "Exiting due to HybridProcessor initialization failure." << std::endl;
        return 1;
    }

    // Set processing mode
    processor.set_processing_mode(processing_mode);
    std::cout << "Processor initialized with '" << processing_mode << "' mode." << std::endl;

    // Process Audio with Enhanced Intelligence
    long num_frames = (audio_data.numel() - frame_size) / hop_size + 1;
    torch::Tensor enhanced_audio = torch::zeros_like(audio_data);
    auto window = torch::hann_window(frame_size);

    // Storage for detailed metrics
    std::vector<ProcessingMetrics> all_metrics;
    all_metrics.reserve(num_frames);

    std::cout << "\nProcessing " << num_frames << " frames with intelligent hybrid system..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < num_frames; ++i)
    {
        // Progress update
        if (i > 0 && i % 100 == 0)
        {
            std::cout << "\r  - Processed frame " << i << "/" << num_frames
                      << " (" << (i * 100 / num_frames) << "%) " << std::flush;
        }

        long start_idx = i * hop_size;

        // Ensure we don't read past the end of the audio buffer
        if (start_idx + frame_size > audio_data.size(0))
        {
            continue;
        }

        auto frame = audio_data.slice(0, start_idx, start_idx + frame_size);

        // Process frame with enhanced intelligence and collect metrics
        ProcessingMetrics metrics = processor.process_frame_with_metrics(frame);
        all_metrics.push_back(metrics);

        // For backward compatibility, also get the processed audio
        auto enhanced_frame = processor.process_frame(frame);

        // Use a window function for smooth overlap-add reconstruction
        if (enhanced_frame.numel() == frame_size)
        {
            enhanced_audio.slice(0, start_idx, start_idx + frame_size) += enhanced_frame * window;
        }

        // Print detailed info for first few frames
        if (i < 3)
        {
            std::cout << "\nFrame " << i << " analysis:" << std::endl;
            std::cout << "  SNR: " << std::fixed << std::setprecision(1) << metrics.estimated_snr << " dB" << std::endl;
            std::cout << "  Noise type: " << metrics.noise_type << std::endl;
            std::cout << "  Strategy: " << metrics.strategy_used << std::endl;
            std::cout << "  DSP weight: " << std::setprecision(2) << metrics.final_dsp_weight << std::endl;
            std::cout << "  ML weight: " << std::setprecision(2) << metrics.final_ml_weight << std::endl;
            std::cout << "  Processing time: " << std::setprecision(2) << metrics.processing_time_ms << " ms" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n\nProcessing completed in " << duration.count() << " ms." << std::endl;
    std::cout << "Average processing time per frame: "
              << (duration.count() / static_cast<double>(num_frames)) << " ms" << std::endl;

    // Calculate and Display Summary Metrics
    SummaryMetrics summary = calculate_summary_metrics(all_metrics);
    print_detailed_analysis(summary);

    // Save Enhanced Output
    save_audio(output_path, enhanced_audio, sample_rate);

    // Save Detailed Metrics
    std::string metrics_filename = output_path.substr(0, output_path.find_last_of('.')) + "_metrics.json";
    save_metrics_to_file(all_metrics, summary, metrics_filename);

    // Performance Assessment
    std::cout << "\nPerformance Assessment:" << std::endl;
    if (summary.avg_processing_time_ms < 10.0)
    {
        std::cout << "Real-time capable (< 10ms per frame)" << std::endl;
    }
    else if (summary.avg_processing_time_ms < 20.0)
    {
        std::cout << "Near real-time (10-20ms per frame)" << std::endl;
    }
    else
    {
        std::cout << "Not real-time ready (> 20ms per frame)" << std::endl;
    }

    // Intelligence Assessment
    if (summary.strategy_distribution.size() > 1)
    {
        std::cout << "Adaptive strategy selection active" << std::endl;
    }
    else
    {
        std::cout << "Limited strategy adaptation" << std::endl;
    }

    if (summary.noise_type_distribution.size() > 1)
    {
        std::cout << "Multiple noise types detected" << std::endl;
    }
    else
    {
        std::cout << "Single noise type environment" << std::endl;
    }

    std::cout << "\nEnhanced hybrid processing complete!" << std::endl;
    std::cout << "Files generated:" << std::endl;
    std::cout << "  - Enhanced audio: " << output_path << std::endl;
    std::cout << "  - Processing metrics: " << metrics_filename << std::endl;

    return 0;
}