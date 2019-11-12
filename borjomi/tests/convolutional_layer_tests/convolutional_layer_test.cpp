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
  ConvolutionalLayerV2 convv2(32, 32, 3, 5, 32, padding::same, true, engine_t::internal);

  size_t batchSize = 10;

  conv.initialize();
  convv2.initialize();

  conv.setBatchSize(batchSize);
  convv2.setBatchSize(batchSize);

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
	matrix_t weightsForConvv2(32, 5*3*5);

	for (size_t rowIdx = 0; rowIdx < weights.rows(); rowIdx++) {
		for (size_t colIdx = 0; colIdx < weights.cols(); colIdx++) {
			weightsForConv.at(0, rowIdx * 5*5*3 + colIdx) = weights.at(rowIdx, colIdx);
		}
	}

  shuffleColAndChannelDimesions(weights, shape3d_t(5,5,3), weightsForConvv2);

  matrix_t bias = generateBias();

  conv.setEdgeData("weight", weightsForConv);
  convv2.setEdgeData("weight", weightsForConvv2);
  conv.setEdgeData("bias", bias);
  convv2.setEdgeData("bias", bias);

  conv.setEdgeData("incomingEdge", data);
  convv2.setEdgeData("incomingEdge", data);

  conv.forward();
  convv2.forward();

  matrix_t convOutput = conv.getEdgeData("outgoingEdge");
  matrix_t convv2Output = convv2.getEdgeData("outgoingEdge");

  std::pair<int, float> compareResult = compare(convOutput, convv2Output);
  std::cout << "Comparing forward outputs:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;

  conv.setEdgeGradient("outgoingEdge", grad);
  convv2.setEdgeGradient("outgoingEdge", grad);

  conv.backward();
  std::cout << "\n\n";
  convv2.backward();

  matrix_t convPrevGrad = conv.getEdgeGradient("incomingEdge");
  matrix_t convv2PrevGrad = convv2.getEdgeGradient("incomingEdge");

  matrix_t convWeightGrad = conv.getEdgeGradient("weight");
  matrix_t convv2WeightGrad = convv2.getEdgeGradient("weight");
  matrix_t convv2WeightGradReordered(convv2WeightGrad.shape());
  shuffleColAndChannelDimesions(convv2WeightGrad, shape3d_t(5,3,5), convv2WeightGradReordered);
  matrix_t convv2WeightGradSingleRow(shape2d_t(1, 5*5*96));

	for (size_t rowIdx = 0; rowIdx < convv2WeightGradReordered.rows(); rowIdx++) {
		for (size_t colIdx = 0; colIdx < convv2WeightGradReordered.cols(); colIdx++) {
			convv2WeightGradSingleRow.at(0, rowIdx * 5*5*3 + colIdx)
        = convv2WeightGradReordered.at(rowIdx, colIdx);
		}
	}

  compareResult = compare(convPrevGrad, convv2PrevGrad);
  std::cout << "Comparing prev gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;

  compareResult = compare(convWeightGrad, convv2WeightGradSingleRow);
  std::cout << "Comparing weight gradients:" << std::endl;
  std::cout << "  Number of differences: " << compareResult.first << std::endl;
  std::cout << "  Max distance: " << compareResult.second << std::endl;
}

int main(int argc, char** argv) {

		std::cout << std::endl << std::endl;
    testConvLayer();
    return 0;
}