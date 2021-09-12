//
// Created by lyx on 9/11/21.
//

#ifndef MNIST_ONNXMNIST_H
#define MNIST_ONNXMNIST_H

#include "params.h"
#include "buffers.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>

class OnnxMnist {
public:
  OnnxMnist(const common::OnnxParams& params);
  bool constructNetwork(std::shared_ptr<nvinfer1::IBuilder>& builder,
                    std::shared_ptr<nvinfer1::INetworkDefinition>& network,
                    std::shared_ptr<nvinfer1::IBuilderConfig>& config,
                    std::shared_ptr<nvonnxparser::IParser>& parser);
   bool build();
   bool infer();
   bool processInput(const samplesCommon::BufferManager& buffers);
   bool verifyOutput(const samplesCommon::BufferManager& buffers);

private:
    common::OnnxParams mParams_;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    int mNumber{0};



};









#endif //MNIST_ONNXMNIST_H
