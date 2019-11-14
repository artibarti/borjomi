#include "borjomi/borjomi.h"

using namespace borjomi;

void shuffleColAndChannelDimesions(const matrix_t& input,
  const shape3d_t& currentShape, matrix_t& reshaped) {

  shape3d_t newShape(currentShape.rows_, currentShape.channels_, currentShape.cols_);
  for (size_t sampleIdx = 0; sampleIdx < input.rows(); sampleIdx++) {
    for (size_t originalChannelIdx = 0; originalChannelIdx < currentShape.channels_; originalChannelIdx++) {
      for (size_t originalRowIdx = 0; originalRowIdx < currentShape.rows_; originalRowIdx++) {
        for (size_t originalColIdx = 0; originalColIdx < currentShape.cols_; originalColIdx++) {
          reshaped.at(sampleIdx, newShape.getIndex(originalRowIdx, originalChannelIdx, originalColIdx))
            = input.at(sampleIdx, currentShape.getIndex(originalRowIdx, originalColIdx, originalChannelIdx));
        }
      }
    }
  }
}

matrix_t generateWeights() {

  size_t fanInSize = 5 * 5 * 3;
  size_t fanOutSize = 5 * 5 * 32;

  matrix_t weight(32, 5 * 5 * 3);
  WeightInitializer::initialize(WeightInitializerType::xavier,
    weight, fanInSize, fanOutSize);

  return weight;
}

matrix_t generateBias() {

  size_t fanInSize = 5 * 5 * 3;
  size_t fanOutSize = 5 * 5 * 32;

  matrix_t bias(1, 32);
  WeightInitializer::initialize(WeightInitializerType::constant,
    bias, fanInSize, fanOutSize);

  return bias;
}

void testConvLayer() {

  ConvolutionalLayer conv(32, 32, 3, 5, 5, 32, padding::same, true, engine_t::internal);
  ConvolutionalXLayer convx(32, 32, 3, 5, 32, padding::same, true, engine_t::internal);

  size_t batchSize = 10;

  conv.initialize();
  convx.initialize();

  conv.setBatchSize(batchSize);
  convx.setBatchSize(batchSize);

  matrix_t data(batchSize, 32 * 32 * 3);
  matrix_t grad(batchSize, 32 * 32 * 32);

  for (size_t idx = 0; idx < data.size(); idx++) {
    data.at(idx) = uniformRand(-2.0, 2.0);
  }

  for (size_t idx = 0; idx < grad.size(); idx++) {
    grad.at(idx) = uniformRand(-2.0, 2.0);
  }

  matrix_t weights = generateWeights();

	matrix_t weightsForConv(1, 5*5*96);
	matrix_t weightsForConvx(32, 5*3*5);

	for (size_t rowIdx = 0; rowIdx < weights.rows(); rowIdx++) {
		for (size_t colIdx = 0; colIdx < weights.cols(); colIdx++) {
			weightsForConv.at(0, rowIdx * 5*5*3 + colIdx) = weights.at(rowIdx, colIdx);
		}
	}

  shuffleColAndChannelDimesions(weights, shape3d_t(5,5,3), weightsForConvx);

  matrix_t bias = generateBias();

  conv.setEdgeData("weight", weightsForConv);
  convx.setEdgeData("weight", weightsForConvx);
  conv.setEdgeData("bias", bias);
  convx.setEdgeData("bias", bias);

  conv.setEdgeData("incomingEdge", data);
  convx.setEdgeData("incomingEdge", data);

  conv.forward();
  convx.forward();

  matrix_t convOutput = conv.getEdgeData("outgoingEdge");
  matrix_t convxOutput = convx.getEdgeData("outgoingEdge");

  std::pair<int, float> compareResult = compare(convOutput, convxOutput);
  std::cout << "Comparing forward outputs:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;

  conv.setEdgeGradient("outgoingEdge", grad);
  convx.setEdgeGradient("outgoingEdge", grad);

  conv.backward();
  std::cout << "\n\n";
  convx.backward();

  matrix_t convPrevGrad = conv.getEdgeGradient("incomingEdge");
  matrix_t convxPrevGrad = convx.getEdgeGradient("incomingEdge");

  matrix_t convWeightGrad = conv.getEdgeGradient("weight");
  matrix_t convxWeightGrad = convx.getEdgeGradient("weight");
  matrix_t convxWeightGradReordered(convxWeightGrad.shape());
  shuffleColAndChannelDimesions(convxWeightGrad, shape3d_t(5,3,5), convxWeightGradReordered);
  matrix_t convxWeightGradSingleRow(shape2d_t(1, 5*5*96));

	for (size_t rowIdx = 0; rowIdx < convxWeightGradReordered.rows(); rowIdx++) {
		for (size_t colIdx = 0; colIdx < convxWeightGradReordered.cols(); colIdx++) {
			convxWeightGradSingleRow.at(0, rowIdx * 5*5*3 + colIdx)
        = convxWeightGradReordered.at(rowIdx, colIdx);
		}
	}

  compareResult = compare(convPrevGrad, convxPrevGrad);
  std::cout << "Comparing prev gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;

  compareResult = compare(convWeightGrad, convxWeightGradSingleRow);
  std::cout << "Comparing weight gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;
}

int main(int argc, char** argv) {

		std::cout << std::endl << std::endl;
    testConvLayer();
    return 0;
}