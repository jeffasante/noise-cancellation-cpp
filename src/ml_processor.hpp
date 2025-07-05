#ifndef ML_PROCESSOR_HPP
#define ML_PROCESSOR_HPP
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

class MLProcessor {
public:
    MLProcessor(const std::string& model_path);
    bool is_loaded() const;
    torch::Tensor process(const torch::Tensor& audio_chunk);
private:
    torch::jit::script::Module module_;
    bool model_loaded_ = false;
};
#endif