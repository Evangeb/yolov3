#include <torch/torch.h>
#include <vector>


struct routeLayerConcat : torch::nn::Module {
    routeLayerConcat(int , int , std::vector<torch::Tensor>&);



    torch::Tensor forward(torch::Tensor);

    int64_t routeDim = 1;
    int sizeVector;
    int fromrouteOne;
    int fromrouteTwo;
    std::vector<torch::Tensor>& layers;
};

