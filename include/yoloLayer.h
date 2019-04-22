#include <torch/torch.h>
#include <vector>


struct yoloLayer : torch::nn::Module {
    yoloLayer(int , int , std::vector<std::pair<float, float> > , std::vector<torch::Tensor>&,torch::Tensor&);


    torch::Tensor forward(torch::Tensor);

    int inputDim;
    std::vector<std::pair<float, float> > anchors;
    int numClasses;
    std::vector<torch::Tensor>& layers;
    torch::Tensor& detections;
};
