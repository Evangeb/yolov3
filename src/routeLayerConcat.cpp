#include <torch/torch.h>
#include <vector>
#include <routeLayerConcat.h>


routeLayerConcat::routeLayerConcat(int routeOne, int routeTwo, std::vector<torch::Tensor> &layer_ref) : layers(layer_ref), fromrouteOne(routeOne), fromrouteTwo(routeTwo) {}



torch::Tensor routeLayerConcat::forward(torch::Tensor x ) {
    std::cout<< "Route Concat Forward Layer" << std::endl;
    sizeVector = layers.size();
    torch::Tensor firstLayer = layers.back();
    torch::Tensor secondLayer = layers[fromrouteTwo];

    x = torch::cat({firstLayer,secondLayer}, routeDim);

    layers.push_back(x);

    return x;
}


