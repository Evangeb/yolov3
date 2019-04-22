#include <torch/torch.h>
#include <vector>

/*
struct routeLayer: torch::nn::Module {
    routeLayer(int from, std::vector<torch::Tensor> &layer_ref) : layers(layer_ref), fromint(from) {}


    torch::Tensor forward(torch::Tensor x ) {
        std::cout<< "Route Forward Layer" << std::endl;
        sizeVector = layers.size();

        x = layers[sizeVector + fromint];

        layers.push_back(x);

        return x;
    }
    int sizeVector;
    int fromint;
    std::vector<torch::Tensor>& layers;

};
*/

struct routeLayer: torch::nn::Module {
    routeLayer(int, std::vector<torch::Tensor>&);


    torch::Tensor forward(torch::Tensor);
    int sizeVector;
    int fromint;
    std::vector<torch::Tensor>& layers;

};
