//
// Created by lyx on 9/11/21.
//

#include "onnxMnist.h"
#include "logger.h"
#include <NvInferRuntime.h>
#include "common.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kERROR) : reportableSeverity(severity) {}
    void log(Severity severity, const char *msg) noexcept {
      // suppress messages with severity enum value greater than the reportable
      if (severity > reportableSeverity) return;
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "INTERNAL_ERROR: ";
          break;
        case Severity::kERROR:
          std::cerr << "ERROR: ";
          break;
        case Severity::kWARNING:
          std::cerr << "WARNING: ";
          break;
        case Severity::kINFO:
          std::cerr << "INFO: ";
          break;
        default:
          std::cerr << "UNKNOWN: ";
          break;
      }
      std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
}gLogger;

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
      delete obj;
    }
};

OnnxMnist::OnnxMnist(const common::OnnxParams &params) :
mParams_(params), mEngine_(nullptr) {
}

bool OnnxMnist::constructNetwork(std::shared_ptr<nvinfer1::IBuilder>& builder,
                             std::shared_ptr<nvinfer1::INetworkDefinition>& network,
                             std::shared_ptr<nvinfer1::IBuilderConfig>& config,
                             std::shared_ptr<nvonnxparser::IParser>& parser) {
  auto parsed = parser->parseFromFile(mParams_.onnxFileName.c_str(), 1);
  if (!parsed) {
    return false;
  }
  std::size_t _MiB = 1 << 20;
  config->setMaxWorkspaceSize(16 * _MiB);
  if (mParams_.fp16) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  //to do
  if (mParams_.int8) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  return true;
}


bool OnnxMnist::build() {
  auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
//auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }
  auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }
  auto parser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
  if (!parser) {
    return false;
  }
  auto constructed = constructNetwork(builder, network, config, parser);
  if (!constructed) {
    return false;
  }
  std::shared_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }
  //std::shared<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  std::shared_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
  if (!runtime) {
    return false;
  }
  mEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr),
                                                                        InferDeleter());
  if (!mEngine_) {
    return false;
  }
  ASSERT(network->getNbInputs() == 1);
  mInputDims = network->getInput(0)->getDimensions();
  ASSERT(mInputDims.nbDims == 4);

  ASSERT(network->getNbOutputs() == 1);
  mOutputDims = network->getOutput(0)->getDimensions();
  ASSERT(mOutputDims.nbDims == 2);

  return true;
}
bool OnnxMnist::infer() {
  samplesCommon::BufferManager buffers(mEngine_);
  auto context = std::shared_ptr<nvinfer1::IExecutionContext>(
          mEngine_->createExecutionContext());
  if (!context) {
    return false;
  }
  ASSERT(mParams_.inputTensorNames.size() == 1);
  if (!processInput(buffers)) {
    return false;
  }
  buffers.copyInputToDevice();
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers.copyOutputToHost();

  if (verifyOutput(buffers)) {
    return false;
  }


  return true;
}
bool OnnxMnist::processInput(const samplesCommon::BufferManager& buffers) {
  const int inputH = mInputDims.d[2];
  const int inputW = mInputDims.d[3];

  srand(unsigned(time(nullptr)));
  std::vector<uint8_t> fileData(inputH * inputW);
  mNumber = rand() % 10;
  readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams_.dataDirs), fileData.data(), inputH, inputW);
  sample::gLogInfo << "Input:" << std::endl;
  for (int i = 0; i < inputH * inputW; ++i) {
    sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
  }
  sample::gLogInfo << std::endl;
  float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams_.inputTensorNames[0]));
  for (int i = 0; i < inputH * inputW; ++i) {
    hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
  }
  return true;
}

bool OnnxMnist::verifyOutput(const samplesCommon::BufferManager& buffers) {
  const int outputSize = mOutputDims.d[1];
  float* output = static_cast<float*>(buffers.getHostBuffer(mParams_.outputTensorNames[0]));
  float val{0.0f};
  int idx{0};

  float sum{0.0f};
  for (int i = 0; i < outputSize; ++i) {
    output[i] = exp(output[i]);
    sum += output[i];
  }
  sample::gLogInfo << "Output:" << std::endl;
  for (int i = 0; i < outputSize; ++i) {
    output[i] /= sum;
    val = std::max(val, output[i]);
    if (val == output[i]) {
      idx = i;
    }
    sample::gLogInfo << "Prob " << i << " " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                     << " "
                     << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                     << std::endl;
  }
  sample::gLogInfo << std::endl;

  return idx == mNumber && val > 0.9f;
}