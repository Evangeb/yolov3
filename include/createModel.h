#include <torch/torch.h>
#include <vector>
#include "utils.h"
#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/stat.h>


torch::nn::Sequential createModel(std::vector<std::map<std::string,std::string> >, torch::Tensor&, std::vector<torch::Tensor>&);
