#ifndef GUARD_utils_h
#define GUARD_utils_h

#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

typedef struct{
    int w,h,c;
    unsigned char *data;
}image_stb;

std::vector<std::map<std::string,std::string> > parseConfig(std::string);
torch::Tensor bboxIOU(torch::Tensor, torch::Tensor);
torch::Tensor NMS(torch::Tensor&);
void draw_boxes(torch::Tensor&, torch::Tensor&, int);
void draw_boxes_raw(unsigned char*, torch::Tensor&, int, int, int);
image_stb load_image(char*, int );
image_stb resize_image(image_stb, float, int);
torch::Tensor image_stb_to_Tensor(image_stb img);
torch::Tensor embed_Tensor_in_letterbox(torch::Tensor, int, int, int, int, int, int);
torch::Tensor format_detections(torch::Tensor&);
torch::Tensor scale_detections(torch::Tensor&, int, int, float);
torch::Tensor output_detections(torch::Tensor&, int, int, float);
void write_image(torch::Tensor, image_stb, int);
#endif
