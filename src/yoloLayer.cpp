#include <torch/torch.h>
#include <vector>
#include <yoloLayer.h>

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}



    yoloLayer::yoloLayer(int inputDim, int numClasses, std::vector<std::pair<float, float> > anchors, std::vector<torch::Tensor> &layer_ref,torch::Tensor& detectionLayer) : layers(layer_ref), numClasses(numClasses), anchors(anchors), inputDim(inputDim), detections(detectionLayer) {}


    torch::Tensor yoloLayer::forward(torch::Tensor x ) {
        std::cout<< "YOLO Forward Layer" << std::endl;
        auto x_a = x.accessor<float,4>();
        int batch_size = x_a.size(0);
        float stride = inputDim/x_a.size(2);
        int gridSize = inputDim/stride;
        int bboxAttributes = 5 + numClasses;
        int numAnchors = anchors.size();


        static bool write = false;
        x = torch::reshape(x,{1,bboxAttributes * numAnchors, gridSize*gridSize});
        x = torch::transpose(x,1,2).contiguous();
        x = torch::reshape(x,{1,gridSize*gridSize*numAnchors, bboxAttributes});
        
        torch::Tensor anchorTensor = torch::rand({numAnchors,2});

        for (int i = 0; i < anchors.size(); ++i)
        {

            anchors[i].first /=  stride;
            anchors[i].second /=  stride;
            anchorTensor[i][0] = anchors[i].first;
            anchorTensor[i][1] = anchors[i].second;
        }


        anchorTensor = anchorTensor.repeat({gridSize*gridSize, 1}).unsqueeze(0);
        torch::Tensor pred_x = torch::sigmoid(x.narrow(2,0,1));

        torch::Tensor pred_y = torch::sigmoid(x.narrow(2,1,1));
        torch::Tensor objectness = torch::sigmoid(x.narrow(2,4,1));
        torch::Tensor pred_hw = x.narrow(2,2,2);
        torch::Tensor pred_xy = torch::cat({pred_x,pred_y}, 2);

        std::vector<int> gridArange = arange(0,gridSize,1);

        torch::Tensor x_offset = torch::rand({gridSize});

        for (int i = 0; i < gridSize; i++)
        {
            x_offset[i] = i;
        }
        x_offset = x_offset.repeat(gridSize);


        torch::Tensor y_offset = torch::rand({gridSize*gridSize});
        
        int count = 0;
        for (int i = 0; i < gridSize*gridSize; i++)
        {
            
            y_offset[i] = count;
            if (i % gridSize == 0 && i != 0)
            {
                count += 1;
            }
        }

        x_offset = x_offset.unsqueeze(1);
        y_offset = y_offset.unsqueeze(1);


        torch::Tensor x_y_offset = torch::cat({x_offset,y_offset}, 1);

        x_y_offset = x_y_offset.repeat({1,numAnchors}).reshape({-1,2}).unsqueeze(0);

        pred_xy += x_y_offset;
        pred_hw = torch::exp(pred_hw)*anchorTensor;

        torch::Tensor pred_classes = torch::sigmoid(x.narrow(2,5,numClasses));
        torch::Tensor pred_coords = torch::cat({pred_xy,pred_hw},2)*stride;

        x = torch::cat({pred_coords,objectness,pred_classes},2);


        if (x.sizes()[1] == 1083) {
            detections = x;
            write = true;
        } else {
            detections = torch::cat({detections,x},1);

        }
        layers.push_back(x);
        return x;

        
    }


