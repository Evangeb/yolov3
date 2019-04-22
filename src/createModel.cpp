#include "upsampleLayer.h"
#include "convLayer.h"
#include "routeLayer.h"
#include "shortCutLayer.h"
#include "routeLayerConcat.h"
#include "yoloLayer.h"
#include "convOutputLayer.h"
#include <torch/torch.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <createModel.h>

struct stat results;

torch::nn::Sequential createModel(std::vector<std::map<std::string,std::string> > layers, torch::Tensor &detections, std::vector<torch::Tensor> &featureMaps){

    torch::nn::Sequential yolo;
    std::map<std::string,std::string> net;
    std::map<std::string,std::string> net_info;
    

    int numWeights;
    if (stat("yolov3.weights", &results) == 0 )
    {
        auto binSize = results.st_size;
        std::cout << binSize << std::endl;
        numWeights = (binSize-20)/4;
        
    } else {
        std::cout << "ERROR" << std::endl;
    }


    int previousFilter = 3;    
    int filters;
    int kernelSize;
    int stride;
    int padding;
    int padSize;
    int batchNorm;
    int upsampleCount = 0;
    int from;
    bool bias;
    float slope = 0.1;
    int upsampleFactor;
    std::vector<int> outputFilters;
    //std::vector<torch::Tensor> featureMaps;
    //torch::Tensor detections;

    char header[20];


    std::ifstream input("yolov3.weights", std::ios::binary);
    input.read(header,20);
    for(int i = 0; i < layers.size(); i++){


        net = layers[i];

        if (net["Layer Type"] == "convolutional") {


            filters = std::stoi(net["filters"]);
            kernelSize = std::stoi(net["size"]);
            stride = std::stoi(net["stride"]);
            padding = std::stoi(net["pad"]);

           
            if( net.count("batch_normalize") ) {
                batchNorm = std::stoi(net["batch_normalize"]);
                bias = false;
            } 
            else {
                batchNorm = 0;
                bias = true;
            }

            if (padding == 1){
                padSize = (kernelSize - 1) / 2;
            } 
            else {
                padSize = 0;
            }

            if (batchNorm) {
                
                convLayer conv(previousFilter, filters, kernelSize, stride, padSize, bias, featureMaps, input);
                yolo->push_back(conv);
            } else {
                
                convOutputLayer conv(previousFilter, filters, kernelSize, stride, padSize, bias, featureMaps, input);
                yolo->push_back(conv);
            }

            previousFilter = filters;

        } 

        else if (net["Layer Type"] == "upsample") {

            upsampleFactor = std::stoi(net["stride"]);
            if (upsampleCount == 0) {
                upSampleLayer upsample(1, 256, 38, 38, true, featureMaps);
                yolo->push_back(upsample);
                upsampleCount = 1;
            } else {
                upSampleLayer upsample(1, 128, 76, 76, true, featureMaps);
                yolo->push_back(upsample);
            }
        }
        else if (net["Layer Type"] == "shortcut") {

            from = std::stoi(net["from"]);
            shortCutLayer shortcut(from, featureMaps);
            yolo->push_back(shortcut);
        }
        else if (net["Layer Type"] == "route") {

            if (net["layers"].find(',') != std::string::npos)
            {

                std::string routingLayers = net["layers"];

                int routeOne = std::stoi(std::string(routingLayers.begin(), routingLayers.begin()+2));
                int routeTwo = std::stoi(std::string(routingLayers.begin()+3, routingLayers.end()));
                auto featureMapSize = static_cast<int>(outputFilters.size());
                int firstFilterSize = outputFilters.back();
                int secondSizeIndex = routeTwo - featureMapSize;
                int secondFiltersize = outputFilters[routeTwo];


                filters = firstFilterSize + secondFiltersize;
                previousFilter = filters;
                routeLayerConcat routeConcat(routeOne, routeTwo, featureMaps);
                yolo->push_back(routeConcat);

            } else {
                int routeInt = std::stoi(net["layers"]);


                auto featureMapSize = static_cast<int>(outputFilters.size());
                filters = outputFilters[featureMapSize + routeInt];
                previousFilter = filters;

                routeLayer route(routeInt, featureMaps);

                yolo->push_back(route);
    
                

            }
            std::cout << std::endl;
        }
        else if (net["Layer Type"] == "yolo") {
            std::vector<float> anchors;
            std::stringstream ss(net["anchors"]);
            while( ss.good() )
            {
                std::string substr;
                std::getline( ss, substr, ',');
                anchors.push_back(std::stof(substr));
            }

            std::vector<int> masks;
            std::stringstream ss2(net["mask"]);
            while( ss2.good() )
            {
                std::string substr;
                std::getline( ss2, substr, ',');
                masks.push_back(std::stoi(substr));
            }


            std::vector<std::pair<float, float> > anchorPairs;

            
            int inputDim = std::stoi(net_info["height"]);
            int numClasses = std::stoi(net["classes"]);
            
            int count = 0;

            for (int i = 0; i < anchors.size(); i += 2)
            {
                if (std::find(masks.begin(), masks.end(), count) != masks.end() ){
                    std::pair<float,float> anchorPair(anchors[i], anchors[i+1]);
                    anchorPairs.push_back(anchorPair);
                }
                count += 1;

            } 

            yoloLayer yoloOut(inputDim, numClasses, anchorPairs, featureMaps, detections);
            yolo->push_back(yoloOut);
        }
        else if (net["Layer Type"] == "net") {
            net_info = net;
        } 
        else {
            std::cout << "Invalid Layer Type" << std::endl;
        }


        if (i != 0){ 
            outputFilters.push_back(filters);
        }

    }
    input.close();
    return yolo;
}
