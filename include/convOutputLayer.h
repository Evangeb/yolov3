#include <torch/torch.h>
#include <vector>


struct convOutputLayer : torch::nn::Module {
    convOutputLayer(int, int, int, int, int, bool, std::vector<torch::Tensor>&, std::ifstream&);

    torch::Tensor forward(torch::Tensor);

    torch::nn::Conv2d conv;
    std::vector<torch::Tensor>& layers;
};
