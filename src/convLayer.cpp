#include <torch/torch.h>
#include <vector>
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
#include <convLayer.h>

template<class T>
void read(std::ifstream &input, std::vector<T> &v){
	input.read( (char *) &v[0], sizeof(T)*v.size() );
}

convLayer::convLayer(int previousFilter, int filters, int kernelSize, int stride, int padSize, bool bias, std::vector<torch::Tensor> &layer_ref, std::ifstream& weightsFile)
        : conv(torch::nn::Conv2d(torch::nn::Conv2dOptions(previousFilter, filters, kernelSize)
                                                        .stride(stride)
                                                        .padding(padSize)
                                                        .with_bias(false))),
          batch_norm(torch::nn::BatchNormOptions(filters).stateful(true).affine(true)),
          layers(layer_ref)
    {

        register_module("conv", conv);
        register_module("batch_norm", batch_norm);
        
        //BATCH NORM SECTION
        
        int numBiasWeights = filters;

        auto batchNormBiasDim = batch_norm->bias.sizes();
        auto batchNormWeightDim = batch_norm->weight.sizes();
        auto batchNormMeanDim = batch_norm->running_mean.sizes();
        auto batchNormVarDim = batch_norm->running_var.sizes();

        std::vector<float> batchNormBiases(numBiasWeights);
        std::vector<float> batchNormWeights(numBiasWeights);
        std::vector<float> batchNormMean(numBiasWeights);
        std::vector<float> batchNormVar(numBiasWeights);      

        read(weightsFile, batchNormBiases);
        read(weightsFile, batchNormWeights);
        read(weightsFile, batchNormMean);
        read(weightsFile, batchNormVar);

        torch::Tensor batchNormBiasesTensor = torch::rand({numBiasWeights});
        torch::Tensor batchNormWeightsTensor = torch::rand({numBiasWeights});
        torch::Tensor batchNormMeanTensor = torch::rand({numBiasWeights});
        torch::Tensor batchNormVarTensor = torch::rand({numBiasWeights});


        auto bias_a = batchNormBiasesTensor.accessor<float,1>();

        for (int i = 0; i < batchNormBiases.size(); i++){
            bias_a[i] = batchNormBiases[i];
        }

        auto weight_a = batchNormWeightsTensor.accessor<float,1>();

        for (int i = 0; i < batchNormWeights.size(); i++){
            weight_a[i] = batchNormWeights[i];
        }

        auto mean_a = batchNormMeanTensor.accessor<float,1>();

        for (int i = 0; i < batchNormMean.size(); i++){
            mean_a[i] = batchNormMean[i];
        }

        auto var_a = batchNormVarTensor.accessor<float,1>();

        for (int i = 0; i < batchNormVar.size(); i++){
            var_a[i] = batchNormVar[i];
        }

        batch_norm->bias = batchNormBiasesTensor.view_as(batch_norm->bias);
        batch_norm->weight = batchNormWeightsTensor.view_as(batch_norm->weight);
        batch_norm->running_mean = batchNormMeanTensor.view_as(batch_norm->running_mean);
        batch_norm->running_var = batchNormVarTensor.view_as(batch_norm->running_var);

        int numConvWeights = previousFilter*filters*kernelSize*kernelSize;


        auto convWeightDims = conv->weight.sizes();

        
        std::vector<float> convWeights(numConvWeights);

        read(weightsFile, convWeights);
        
        torch::Tensor convWeightsTensor = torch::rand({numConvWeights});
        
        auto conv_a = convWeightsTensor.accessor<float,1>();

        for (int i = 0; i < convWeights.size(); i++){
            conv_a[i] = convWeights[i];
        }


        conv->weight = convWeightsTensor.view_as(conv->weight);
    }


    torch::Tensor convLayer::forward(torch::Tensor x) {

        std::cout<< "Conv Forward Layer" << std::endl;
        //std::cout << "Before X" << std::endl; 
        //getchar();

        x = torch::leaky_relu(batch_norm(conv(x)),0.1);
        //auto x_a = x.accessor<float,4>();
        //std::cout << "after X" << std::endl;
        //getchar();
        layers.push_back(x);
        //std::cout << "after push back" << std::endl;
        //getchar();
        return x; 
    }


    



