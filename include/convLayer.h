#include <torch/torch.h>
#include <iostream>
#include <string>
#include <map>
#include <iostream>
#include "utils.h"
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <iterator>
#include <stdint.h>
#include <vector>

struct convLayer : torch::nn::Module {
    convLayer(int , int , int , int , int , bool , std::vector<torch::Tensor>& , std::ifstream&);

    torch::Tensor forward(torch::Tensor);

    torch::nn::Conv2d conv;
    torch::nn::BatchNorm batch_norm;
    std::vector<torch::Tensor>& layers;
    

};
