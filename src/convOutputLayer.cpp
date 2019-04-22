#include <torch/torch.h>
#include <vector>
#include <convOutputLayer.h>


template<class T>
void read(std::ifstream &input, std::vector<T> &v){
	input.read( (char *) &v[0], sizeof(T)*v.size() );
}


    convOutputLayer::convOutputLayer(int previousFilter, int filters, int kernelSize, int stride, int padSize, bool bias, std::vector<torch::Tensor> &layer_ref, std::ifstream& weightsFile)
        : conv(torch::nn::Conv2d(torch::nn::Conv2dOptions(previousFilter, filters, kernelSize)
                                                        .stride(stride)
                                                        .padding(padSize)
                                                        .with_bias(bias))),
          layers(layer_ref)
    {
        
        int numBiases = filters;
        auto biasDims = conv->bias.sizes();
        auto numBias = conv->bias.numel();

        std::vector<float> biasWeights(numBiases);
        
        read(weightsFile, biasWeights);
        
        torch::Tensor convBiasesTensor = torch::rand({numBias});
        auto conv_bias_a = convBiasesTensor.accessor<float,1>();

        for (int i = 0; i < biasWeights.size(); i++){
            conv_bias_a[i] = biasWeights[i];
        }        



        conv->bias = convBiasesTensor.view_as(conv->bias);
        
        auto convWeightDims = conv->weight.sizes();
        auto numConvWeights = conv->weight.numel();

        std::vector<float> convWeights(numConvWeights);
        
        read(weightsFile, convWeights);


        torch::Tensor convWeightsTensor = torch::rand({numConvWeights});
        auto conv_weight_a = convWeightsTensor.accessor<float,1>();

        for (int i = 0; i < convWeights.size(); i++){
            conv_weight_a[i] = convWeights[i];
        }        


        conv->weight = convWeightsTensor.view_as(conv->weight);
        
        register_module("conv", conv);

    }




torch::Tensor convOutputLayer::forward(torch::Tensor x) {
        std::cout<< "Linear Conv Forward Layer" << std::endl;
        x = conv->forward(x);
        layers.push_back(x);   
        return x;
}

