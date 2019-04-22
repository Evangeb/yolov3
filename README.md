## Installation
- Install [Pytorch C++](http://pytorch.org/) select appropriate settings for your environment.
- Clone the repository.
- To build run the following from terminal.
```Shell
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
cd ..
```
where `/path/to/libtorch` should be the path to the unzipped *LibTorch*
distribution, which you can get from the [PyTorch
homepage](https://pytorch.org/get-started/locally/).

- Download yolov3 weights.
```Shell
chmod +x scripts/download_weights.sh
scripts/download_weights.sh
```
- Run detections on single image.
