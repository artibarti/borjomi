#include "borjomi/borjomi.h"

using namespace borjomi;

class ConvXTestResult {

 private:
  matrix_t output_;
  matrix_t prevDelta_;
  matrix_t biasDelta_;
  matrix_t weightDelta_;

 public:
  ConvXTestResult(const matrix_t& output, const matrix_t& prevDelta,
    const matrix_t& weightDelta, const matrix_t& biasDelta) {

    output_ = output;
    prevDelta_ = prevDelta;
    weightDelta_ = weightDelta;
    biasDelta_ = biasDelta;
  }

  const matrix_t& output() const {
    return output_;
  }

  const matrix_t& prevDelta() const {
    return prevDelta_;
  }

  const matrix_t& weightDelta() const {
    return weightDelta_;
  }

  const matrix_t& biasDelta() const {
    return biasDelta_;
  }
};

void generateWeights(matrix_t& weight,
  size_t fanInSize, size_t fanOutSize) {

  WeightInitializer::initialize(WeightInitializerType::xavier,
    weight, fanInSize, fanOutSize);
}

void generateBias(matrix_t& bias,
  size_t fanInSize, size_t fanOutSize) {

  WeightInitializer::initialize(WeightInitializerType::constant,
    bias, fanInSize, fanOutSize);
}

void generateInput(matrix_t& input) {
  for (size_t idx = 0; idx < input.size(); idx++) {
    input.at(idx) = uniformRand(-2.0, 2.0);
  }
}

void generateGrads(matrix_t& grads) {
  for (size_t idx = 0; idx < grads.size(); idx++) {
    grads.at(idx) = uniformRand(-2.0, 2.0);
  }
}

ConvXTestResult runLayer(ConvolutionalXLayer& layer, size_t batchSize, const matrix_t& input,
  const matrix_t& grads, const matrix_t& weights, const matrix_t& bias) {
  
  layer.initialize();
  layer.setBatchSize(batchSize);
  layer.setEdgeData("incomingEdge", input);
  layer.setEdgeData("weight", weights);
  layer.setEdgeData("bias", bias);
  layer.setEdgeGradient("outgoingEdge", grads);

  layer.forward();
  layer.backward();

  return ConvXTestResult(layer.getEdgeData("outgoingEdge"), layer.getEdgeGradient("incomingEdge"),
    layer.getEdgeGradient("weight"), layer.getEdgeGradient("bias"));
}

void compareResults(ConvXTestResult& resultOne, ConvXTestResult& resultTwo) {

  std::pair<int, float> compareResultOutput = compare(resultOne.output(), resultTwo.output());
  std::cout << "Comparing outputs:" << std::endl;
  std::cout << "  Number of differences: " << compareResultOutput.first << std::endl;
  std::cout << "  Max distance: " << compareResultOutput.second << std::endl;

  std::pair<int, float> compareResultPrevGrad = compare(resultOne.prevDelta(), resultTwo.prevDelta());
  std::cout << "Comparing previous gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResultPrevGrad.first << std::endl;
  std::cout << "  Max distance: " << compareResultPrevGrad.second << std::endl;

  std::pair<int, float> compareResultWeightGrad = compare(resultOne.weightDelta(), resultTwo.weightDelta());
  std::cout << "Comparing weight gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResultWeightGrad.first << std::endl;
  std::cout << "  Max distance: " << compareResultWeightGrad.second << std::endl;

  std::pair<int, float> compareResultBiasGrad = compare(resultOne.biasDelta(), resultTwo.biasDelta());
  std::cout << "Comparing bias gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResultBiasGrad.first << std::endl;
  std::cout << "  Max distance: " << compareResultBiasGrad.second << std::endl;
}

int main(int argc, char** argv) {

  shape3d_t inShape(32,32,3);
  shape3d_t outShape(32,32,32);
  shape3d_t weightShape(5,3,5);
  shape3d_t biasShape(32,1,1);

  size_t batchSize = 10;
  size_t fanInSize = 5 * 5 * 3;
  size_t fanOutSize = 5 * 5 * 32;

  matrix_t weights(32, weightShape.size_);
  matrix_t bias(1, biasShape.size_);
  matrix_t input(batchSize, inShape.size_);
  matrix_t grads(batchSize, outShape.size_);

  generateInput(input);
  generateGrads(grads);
  generateWeights(weights, fanInSize, fanOutSize);
	generateBias(bias, fanInSize, fanOutSize);

  ConvolutionalXLayer convxInternal(32, 32, 3, 5, 32, padding::same, true, engine_t::internal);
  ConvolutionalXLayer convxAvx(32, 32, 3, 5, 32, padding::same, true, engine_t::avx);

  ConvXTestResult resultInternal = runLayer(convxInternal, batchSize, input, grads, weights, bias);
  ConvXTestResult resultAvx = runLayer(convxAvx, batchSize, input, grads, weights, bias);

  compareResults(resultInternal, resultAvx);

  std::cout << std::endl << std::endl;
  return 0;
}