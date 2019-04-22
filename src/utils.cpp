#include <fstream>
#include <string>
#include <iostream>
#include <ios>
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>
#include "utils.h"
#include <torch/torch.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*

This reads in the yolov3 configuration file and returns a vector of maps to each layers parameters 

*/

void write_image(torch::Tensor img, image_stb imgOrig, int inputDim){
    
    img = torch::reshape(img, {1*imgOrig.c*inputDim*inputDim});

    auto img_write_a = img.accessor<float,1>();

    unsigned char write_data[inputDim*inputDim*imgOrig.c];

    for (int kk = 0; kk<imgOrig.c;++kk){
        for(int iii = 0; iii < inputDim*inputDim; ++iii){
            write_data[iii*imgOrig.c+kk] = (unsigned char) roundf((255*img_write_a[iii +kk*inputDim*inputDim]));

        }
    }

    stbi_write_png("output.png", imgOrig.w, imgOrig.h, imgOrig.c, imgOrig.data, imgOrig.w*imgOrig.c);

}

torch::Tensor scale_detections(torch::Tensor &detections, int dw, int dh, float ratio){

    auto imageDetectionsX1 = detections.narrow(1,0,1);
    auto imageDetectionsY1 = detections.narrow(1,1,1);
    auto imageDetectionsX2 = detections.narrow(1,2,1);
    auto imageDetectionsY2 = detections.narrow(1,3,1);
    auto imageDetectionsScores = detections.narrow(1,4,3);

    imageDetectionsX1 = (imageDetectionsX1 - dw)/ratio;
    imageDetectionsY1 = (imageDetectionsY1 - dh)/ratio;
    imageDetectionsX2 = (imageDetectionsX2 - dw)/ratio;
    imageDetectionsY2 = (imageDetectionsY2 - dh)/ratio;

    auto imageDectectionsScaled = torch::cat({imageDetectionsX1,imageDetectionsY1,imageDetectionsX2,imageDetectionsY2,imageDetectionsScores},1);

    return imageDectectionsScaled;
    
}

torch::Tensor format_detections(torch::Tensor &detections){

    auto detections_a= detections.accessor<float,3>();
    auto confidences = detections.narrow(2,4,1);

    auto conf_mask = confidences > 0.9;


    auto predictions = detections*conf_mask.toType(at::kFloat);

    auto box_corner1 = predictions.narrow(2,0,1) - (predictions.narrow(2,2,1)/2);
    auto box_corner2 = predictions.narrow(2,1,1) - (predictions.narrow(2,3,1)/2);
    auto box_corner3 = predictions.narrow(2,0,1) + (predictions.narrow(2,2,1)/2);
    auto box_corner4 = predictions.narrow(2,1,1) + (predictions.narrow(2,3,1)/2);
    auto objectness = predictions.narrow(2,4,1);

    auto classes = predictions.narrow(2,5,80);

    auto maxConfidenceTuple = torch::max(classes, 2);
    auto maxConfidence = std::get<0>(maxConfidenceTuple).unsqueeze(2).toType(at::kFloat);
    auto maxConfidenceScore = std::get<1>(maxConfidenceTuple).unsqueeze(2).toType(at::kFloat);
    auto imagePrediction = torch::cat({box_corner1,box_corner2, box_corner3, box_corner4, objectness, maxConfidence, maxConfidenceScore},2);
    auto non_zero_ind = torch::nonzero(objectness.squeeze());
    auto imageDetections = torch::index(imagePrediction.squeeze(0), non_zero_ind);

    imageDetections = imageDetections.squeeze(1);



    return imageDetections;

}

torch::Tensor embed_Tensor_in_letterbox(torch::Tensor img, int dw, int dh, int height, int width, int channels, int inputDim){

    auto letterboxImage = torch::ones({1,channels,inputDim,inputDim})*0.5;


    auto letterboxImage_a = letterboxImage.accessor<float,4>();
    auto img_a = img.accessor<float,4>();
    
    int x,y,q;
  
    for(q = 0; q < channels; ++q) {
        for(y = 0; y < height; ++y){
            for(x = 0; x < width; ++x){
                letterboxImage_a[0][q][y+dh][x+dw] = img_a[0][q][y][x];
        
            }
        }
    }

    return letterboxImage;
}


torch::Tensor image_stb_to_Tensor(image_stb img){

    torch::Tensor imgTensor = torch::ones({img.w*img.h*img.c});
    
    auto imgTensor_a = imgTensor.accessor<float,1>();

    int i,j,k;

    for(k = 0; k < img.c; ++k){
        for(j = 0; j < img.h; ++j){
            for(i = 0; i < img.w; ++i){
                int dst_index = i + img.w*j + img.w*img.h*k;
                int src_index = k + img.c*i + img.c*img.w*j;
                imgTensor_a[dst_index] = (float)img.data[src_index]/255.;
            }
        }
    }

    imgTensor = torch::reshape(imgTensor,{1,img.c,img.h,img.w});

    return imgTensor;
}


image_stb load_image(char* imgName, int channels){
    
    image_stb img;
    int w,h,c;
    unsigned char *data = stbi_load(imgName, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \nSTB Reason: %s\n", stbi_failure_reason());
        exit(0);
    }
    img.w = w;
    img.h = h;
    img.c = c;
    img.data = data;
    
    return img;

}


image_stb resize_image(image_stb img, float ratio, int inputDim){

    image_stb img_resize;
    
    int new_width = (int)(img.w*ratio);
    int new_height = (int)(img.h*ratio);
    int dw = (inputDim - (int)(img.w*ratio))/2;
    int dh = (inputDim - (int)(img.h*ratio))/2;    

    unsigned char resized_data[new_height*new_width*img.c];

    stbir_resize_uint8(img.data ,img.w, img.h , img.w*img.c, resized_data,new_width,new_height, new_width*img.c,img.c);
    
    img_resize.w = new_width;
    img_resize.h = new_height;
    img_resize.c = img.c;
    img_resize.data = resized_data;

    return img_resize;
}

torch::Tensor bboxIOU(torch::Tensor box1, torch::Tensor box2)
{

    auto b1_x1 = box1[0];
    auto b1_y1 = box1[1];
    auto b1_x2 = box1[2];
    auto b1_y2 = box1[3];

    auto b2_x1 = box2.narrow(1,0,1);
    auto b2_y1 = box2.narrow(1,1,1);
    auto b2_x2 = box2.narrow(1,2,1);
    auto b2_y2 = box2.narrow(1,3,1);
    
    auto interRectx1 = torch::max(b1_x1, b2_x1);
    auto interRecty1 = torch::max(b1_y1, b2_y1);
    auto interRectx2 = torch::min(b1_x2, b2_x2);
    auto interRecty2 = torch::min(b1_y2, b2_y2);

    auto interArea = torch::clamp(interRectx2 - interRectx1 + 1, 0) * torch::clamp(interRecty2 - interRecty1 + 1, 0);

    auto b1Area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
    auto b2Area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);
    
    auto iou = interArea / (b1Area + b2Area - interArea);

    return iou;

}

torch::Tensor NMS(torch::Tensor &detections)
{

    torch::Tensor detectionsNMS;
    auto classes = detections.narrow(1,6,1);
    auto uniqueClassesTuple = torch::_unique(classes);
    auto uniqueClasses = std::get<0>(uniqueClassesTuple);
    auto uniqueClasses_a = uniqueClasses.accessor<float,1>();
    
    int numUniqueClasses = uniqueClasses.sizes()[0];

    for( int i = 0; i < numUniqueClasses; i++)
    {
        int currentClass = uniqueClasses_a[i];
        auto classMask = classes == currentClass;
        auto currentDetections = detections*classMask.toType(at::kFloat);
        auto nonZeroDetections = torch::nonzero(classMask.squeeze()); 
        auto classDetections = torch::index(currentDetections,nonZeroDetections).squeeze(1);
        auto objectness = classDetections.narrow(1,4,1);
        auto classOrder = objectness.argsort(0,true);
        auto orderedClassDetection = torch::index(classDetections,classOrder).squeeze(1);      
        auto numClassDetections = orderedClassDetection.sizes()[0];
        
        for( int j = 0; j < numClassDetections; j++)
        {
            torch::Tensor ious;

            auto maxSize = orderedClassDetection.sizes()[0];

            if (orderedClassDetection.sizes()[0]-1 < j) break;
            if (orderedClassDetection.sizes()[0] == 1) break;

            ious = bboxIOU(orderedClassDetection[j], orderedClassDetection.narrow(0,j+1,(maxSize-1-j)));
            auto iouMask = (ious < 0.5).toType(at::kFloat);

            auto zeroMasked = orderedClassDetection.narrow(0,j+1,(maxSize-1-j))*iouMask;
            auto maskedOrderedClassDetections = torch::cat({orderedClassDetection.narrow(0,0,j+1),zeroMasked},0);
            auto nonZero = torch::nonzero(maskedOrderedClassDetections.narrow(1,0,1).squeeze());
            auto validDetections = torch::index(maskedOrderedClassDetections,nonZero).squeeze(1);


            orderedClassDetection = validDetections;

        }

        if (i == 0){
            detectionsNMS = orderedClassDetection;
        } else {
            detectionsNMS = torch::cat({detectionsNMS, orderedClassDetection},0);
        }


    }

    return detectionsNMS;

}


torch::Tensor output_detections(torch::Tensor &detections, int dw, int dh, float ratio){


    detections = format_detections(detections);
    detections = NMS(detections);
    detections = scale_detections(detections, dw, dh, ratio);

    return detections;

}

void draw_boxes(torch::Tensor &im, torch::Tensor &detections, int inputDim)
{
    auto detDims = detections.sizes();
    int numDets = detDims[0]; 
    auto im_a = im.accessor<float,4>();
    auto detections_a = detections.accessor<float,2>();

    float r = 0;
    float b = 0;
    float g = 1;


    for (int j = 0; j < numDets; j++) {
        int x1 = detections_a[j][0];
        int y1 = detections_a[j][1];
        int x2 = detections_a[j][2];
        int y2 = detections_a[j][3];

        if(x1 < 0) x1 = 0;
        if(x1 >= inputDim) x1 = inputDim-1;
        if(x2 < 0) x2 = 0;
        if(x2 >= inputDim) x2 = inputDim-1;

        if(y1 < 0) y1 = 0;
        if(y1 >= inputDim) y1 = inputDim-1;
        if(y2 < 0) y2 = 0;
        if(y2 >= inputDim) y2 = inputDim-1;
        
        int i;

        for(i = x1; i <= x2; ++i){
            im_a[0][0][y1][i] = r;
            im_a[0][0][y2][i] = r;

            im_a[0][1][y1][i] = g;
            im_a[0][1][y2][i] = g;

            im_a[0][2][y1][i] = b;
            im_a[0][2][y2][i] = b;
        }

    }
}

void draw_boxes_raw(unsigned char *im, torch::Tensor &detections, int w, int h, int channels)
{
    auto detDims = detections.sizes();
    int numDets = detDims[0]; 
    auto detections_a = detections.accessor<float,2>();

    float r = 0;
    float b = 0;
    float g = 1;


    for (int j = 0; j < numDets; j++) {
        int x1 = detections_a[j][0];
        int y1 = detections_a[j][1];
        int x2 = detections_a[j][2];
        int y2 = detections_a[j][3];

        if(x1 < 0) x1 = 0;
        if(x1 >= w) x1 = w-1;
        if(x2 < 0) x2 = 0;
        if(x2 >= w) x2 = w-1;

        if(y1 < 0) y1 = 0;
        if(y1 >= h) y1 = h-1;
        if(y2 < 0) y2 = 0;
        if(y2 >= h) y2 = h-1;
        
        int i;
        
        for(i = x1; i <= x2; ++i){
            im[0 + channels*i + channels*w*y1] = (unsigned char) roundf(255*r);
            im[0 + channels*i + channels*w*y2] = (unsigned char) roundf(255*r);

            im[1 + channels*i + channels*w*y1] = (unsigned char) roundf(255*g);
            im[1 + channels*i + channels*w*y2] = (unsigned char) roundf(255*g);


            im[2 + channels*i + channels*w*y1] = (unsigned char) roundf(255*b);
            im[2 + channels*i + channels*w*y2] = (unsigned char) roundf(255*b);
        }
        
        for(i = y1; i <= y2; ++i){
            im[0 + channels*x1 + channels*w*i] = (unsigned char) roundf(255*r);
            im[0 + channels*x2 + channels*w*i] = (unsigned char) roundf(255*r);

            im[1 + channels*x1 + channels*w*i] = (unsigned char) roundf(255*g);
            im[1 + channels*x2 + channels*w*i] = (unsigned char) roundf(255*g);


            im[2 + channels*x1 + channels*w*i] = (unsigned char) roundf(255*b);
            im[2 + channels*x2 + channels*w*i] = (unsigned char) roundf(255*b);
        }
        
        
    }
}


std::vector<std::map<std::string,std::string> > parseConfig(std::string configText)
{
    
    std::ifstream configFile(configText.c_str());
    std::string str;
    std::string fileContents;

    while (std::getline(configFile, str))
    {
         
        if (str[0] != '#'){
            fileContents += str;
            fileContents.push_back('\n');
        } else {

        }
    }

    size_t pos;
    
    while ((pos = fileContents.find("\n\n", 0)) != std::string::npos)
    {
        fileContents.erase(pos,1);
    }


    const char &start_delim = '[';
    unsigned last = 0;
    unsigned first = 0;

    std::string fileContentsBlock; 


    std::vector<std::map<std::string,std::string> > layers;

    while (fileContents.find('[',first) != std::string::npos)
    {
        first = last + 1;
        last = fileContents.find('[',first);
        
        fileContentsBlock = fileContents.substr(first, last - first);

        std::istringstream fileContentsBlockStream(fileContentsBlock);
        std::string currentLine;

        bool isName = 1;

        std::map<std::string, std::string> currentLayer;
 
        std::string parameterName;
        std::string parameterValue;
        unsigned positionOfEqual;

        while (std::getline(fileContentsBlockStream, currentLine)) {
            if (isName){
                std::string layerType;
                layerType = currentLine.substr(0, currentLine.size()-1);
                currentLayer["Layer Type"] = layerType;
                isName = 0;
            } else {
                positionOfEqual = currentLine.find('=');
                parameterName = currentLine.substr(0,positionOfEqual);
                std::string::iterator end_pos_name = std::remove(parameterName.begin(), parameterName.end(), ' ');
                parameterName.erase(end_pos_name, parameterName.end()); 


                parameterValue = currentLine.substr(positionOfEqual+1,currentLine.size()-parameterName.size());

                std::string::iterator end_pos = std::remove(parameterValue.begin(), parameterValue.end(), ' ');
                parameterValue.erase(end_pos, parameterValue.end());

                
                currentLayer[parameterName] = parameterValue;
            }
        
        }
        layers.push_back(currentLayer);

    }


    return layers;

}
