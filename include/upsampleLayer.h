#include <torch/torch.h>
#include <vector>

struct upSampleLayer : torch::nn::Module{
    upSampleLayer(int, int , int , int , bool , std::vector<torch::Tensor>&);
    torch::Tensor forward(torch::Tensor);

    torch::nn::Functional upsample;
    std::vector<torch::Tensor>& layers;
};
