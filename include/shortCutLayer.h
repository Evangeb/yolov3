#include <torch/torch.h>
#include <vector>


struct shortCutLayer : torch::nn::Module {
    shortCutLayer(int from, std::vector<torch::Tensor> &layer_ref);
    
    torch::Tensor forward(torch::Tensor x);

    int sizeVector;
    int fromint;
    std::vector<torch::Tensor>& layers;
    torch::Tensor fromLayer;
    torch::Tensor currentLayer;

};
