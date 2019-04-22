#include <torch/torch.h>


#include <string>
#include <map>
#include <iostream>
#include "utils.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <iterator>
#include <stdint.h>

#include "createModel.h"


int main(int argc, char** argv)
{
    
    std::string configText = "cfg/yolov3.cfg";
    std::vector<std::map<std::string,std::string> > layers;
    
    layers = parseConfig(configText);
    auto net_info = layers[0];
 
    int inputDim = std::stoi(net_info["width"]);
    int inputChannels = std::stoi(net_info["channels"]);
    torch::Tensor detections;

    std::vector<torch::Tensor> featureMaps;
    torch::nn::Sequential yolo = createModel(layers, detections,featureMaps);

    image_stb imgOrig = load_image(argv[1], inputChannels);

    float ratio = float(inputDim)/std::max(imgOrig.h,imgOrig.w);
    int dw = (inputDim - (int)(imgOrig.w*ratio))/2;
    int dh = (inputDim - (int)(imgOrig.h*ratio))/2;

    image_stb imgResize = resize_image(imgOrig, ratio, inputDim);
    torch::Tensor img = image_stb_to_Tensor(imgResize);
    img = embed_Tensor_in_letterbox(img , dw, dh, imgResize.h, imgResize.w, imgResize.c, inputDim);

    yolo->eval();
    yolo->forward(img);

    auto imageDetections = output_detections( detections, dw, dh, ratio);

    draw_boxes_raw(imgOrig.data, imageDetections, imgOrig.w, imgOrig.h, inputChannels);

    write_image(img, imgOrig, inputDim);

    return 0;
}
