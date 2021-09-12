#include <iostream>
#include "onnxMnist.h"
#include "params.h"


common::OnnxParams initializeParams(const common::Args &args) {
  common::OnnxParams params;
  if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
  {
    params.dataDirs.push_back("/home/lyx/Downloads/TensorRT-8.0.1.6/data/mnist/");
    params.dataDirs.push_back("/home/lyx/Downloads/TensorRT-8.0.1.6/data/samples/mnist/");
  }
  else //!< Use the data directory provided by the user
  {
    params.dataDirs = args.dataDirs;
  }
  params.onnxFileName = "/home/lyx/Downloads/TensorRT-8.0.1.6/data/mnist/mnist.onnx";
  params.inputTensorNames.push_back("Input3");
  params.outputTensorNames.push_back("Plus214_Output_0");
  params.dlaCore = args.useDLACore;
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;

  return params;
}



int main(int argc, char** argv) {
  common::Args args;
  OnnxMnist mnist(initializeParams(args));
  if (!mnist.build()) {
    std::cout << args.batch << std::endl;
  }
  if (!mnist.infer()) {
    std::cout << "OOOOOOOOOO" << std::endl;
  }


  return 0;
}
