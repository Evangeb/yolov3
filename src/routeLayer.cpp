#include <torch/torch.h>
#include <vector>
#include <routeLayer.h>


routeLayer::routeLayer(int from, std::vector<torch::Tensor>& layer_ref) : layers(layer_ref), fromint(from) 
    {
        
    }

torch::Tensor routeLayer::forward(torch::Tensor x){
        std::cout<< "Route Forward Layer" << std::endl;
        sizeVector = layers.size();

        x = layers[sizeVector + fromint];

        layers.push_back(x);

        return x;

}
