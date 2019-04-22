#include <torch/torch.h>
#include <vector>
#include <upsampleLayer.h>



upSampleLayer::upSampleLayer(int dim, int prevFilters, int x, int y, bool align, std::vector<torch::Tensor> &layer_ref)
         : upsample(torch::nn::Functional(torch::upsample_bilinear2d,(dim,prevFilters,x,y),align)),
           layers(layer_ref)
    {
        //register_module("upsample", upsample);
    }

torch::Tensor upSampleLayer::forward(torch::Tensor x) {
        std::cout<< "Upsample Forward Layer" << std::endl;
        x = upsample->forward(x);
        layers.push_back(x);
        return x;
}
