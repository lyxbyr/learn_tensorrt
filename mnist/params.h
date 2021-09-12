//
// Created by lyx on 9/11/21.
//

#ifndef MNIST_PARAMS_H
#define MNIST_PARAMS_H

#include <vector>
#include <string>

namespace common {

struct Params {
  int32_t batchSize{1};
  int32_t dlaCore{1};
  bool int8{false};
  bool fp16{false};
  std::vector<std::string> dataDirs;
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
};

struct OnnxParams : public Params {
  std::string onnxFileName;
};

struct Args {
  bool runInInt8{false};
  bool runInFp16{false};
  bool help{false};
  int32_t  useDLACore{-1};
  int32_t  batch{1};
  std::vector<std::string> dataDirs;
  std::string saveEngine;
  std::string loadEngine;
  bool useILoop{false};
};






} // namespace common









#endif //MNIST_PARAMS_H
