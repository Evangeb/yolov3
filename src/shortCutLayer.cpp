#include <torch/torch.h>
#include <vector>
#include <shortCutLayer.h>


shortCutLayer::shortCutLayer(int from, std::vector<torch::Tensor> &layer_ref) : layers(layer_ref),fromint(from) {}

torch::Tensor shortCutLayer::forward(torch::Tensor x) {
        std::cout<< "Short Cut Forward Layer" << std::endl;
        sizeVector = layers.size();
        //TODO may be off by one here
        int fromIndex = sizeVector + fromint;
        fromLayer = layers[fromIndex];
        currentLayer = layers.back();
        x = currentLayer + fromLayer;
        layers.push_back(x);
        return x;
    }
