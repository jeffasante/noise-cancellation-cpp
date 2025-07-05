#include "ml_processor.hpp"
#include <iostream>

MLProcessor::MLProcessor(const std::string& model_path) {
    try {
        module_ = torch::jit::load(model_path);
        module_.to(torch::kCPU);
        module_.eval();
        model_loaded_ = true;
        std::cout << "ML model loaded successfully from: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the ML model: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}
bool MLProcessor::is_loaded() const { return model_loaded_; }
torch::Tensor MLProcessor::process(const torch::Tensor& audio_chunk) {
    if (!model_loaded_) return torch::empty({0});
    auto stft = torch::stft(audio_chunk, 1024, 512, 1024, torch::hann_window(1024), false, true, true);
    auto mag_spec = torch::abs(stft).slice(0, 0, 64);
    if (mag_spec.size(1) < 188) {
        mag_spec = torch::nn::functional::pad(mag_spec, torch::nn::functional::PadFuncOptions({0, 188 - mag_spec.size(1)}));
    }
    mag_spec = mag_spec.slice(1, 0, 188);
    auto input_tensor = mag_spec.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs{input_tensor};
    return module_.forward(inputs).toTensor().squeeze(0);
}