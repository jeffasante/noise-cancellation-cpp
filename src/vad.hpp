#ifndef VAD_HPP
#define VAD_HPP
#include <torch/torch.h>
#include <vector>

class SimpleVoiceDetector {
public:
    SimpleVoiceDetector(int sample_rate, int frame_size, int hop_size);
    bool detect_voice(const torch::Tensor& audio_frame);
private:
    double calculate_spectral_centroid(const torch::Tensor& audio_frame);
    int sample_rate_, frame_size_, hop_size_;
    double energy_threshold_ = 0.01;
    double min_speech_freq_ = 200.0, max_speech_freq_ = 4000.0;
};
#endif