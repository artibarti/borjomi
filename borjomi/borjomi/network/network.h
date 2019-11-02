#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <utility>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"
#include "borjomi/network/layer_sequence.h"
#include "borjomi/network/result.h"

namespace borjomi {

class Network {

 private:
  LayerSequence layers_;

  template<LossFunctionType lossFuncType, typename Optimizer>
  void fitBatch(Optimizer& optimizer, const matrix_t& inputs,
    const matrix_t& targets, size_t batchIndexFrom, size_t batchIndexTo);

  matrix_t forwardProp(const matrix_t& inputData);
  
  template<LossFunctionType lossFuncType>
  void backProp(const matrix_t& output, const matrix_t& target);

  std::pair<float, float> getTargetValueRange() const;

 public:

  template<typename Layer>
  Network& operator<< (Layer &&layer);

  LayerSequence* getLayerSequence();

  void updateWeights(Optimizer *optimizer);

  matrix_t predict(const matrix_t& input);

  template<LossFunctionType lossFuncType, typename Optimizer>
  bool fit(Optimizer &optimizer, const matrix_t& inputs, const matrix_t& desiredOutputs, size_t batchSize = 1, int epoch = 1);

  template<LossFunctionType lossFuncType, typename Optimizer,typename OnBatchEnumerate,typename OnEpochEnumerate>
  bool fit(Optimizer &optimizer, const matrix_t& inputs, const std::vector<label_t> &classLabels,
    size_t batchSize, size_t numberOfEpochs, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate);

  template<LossFunctionType lossFuncType, typename Optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
  bool fit(Optimizer &optimizer, const matrix_t& inputs, const matrix_t& targets,
    size_t batchSize, size_t numberOfEpochs, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate);

  Result test(const matrix_t& in, const std::vector<label_t> &t);

  template<LossFunctionType lossFuncType>
  float getLoss(const matrix_t& in, const matrix_t& t);

  template<LossFunctionType lossFuncType>
  float getLoss(const matrix_t& in, const std::vector<label_t> &t);

  size_t getOutDataSize() const;
  size_t getInDataSize() const;

};

template<typename Layer>
Network& Network::operator<< (Layer&& layer) {
  layers_.addLayer(std::forward<Layer>(layer));
  return *this;
}

template<LossFunctionType lossFuncType>
void Network::backProp(const matrix_t& output, const matrix_t& target) {

  matrix_t delta(output.shape());
  LossCalculator::getLossMatrix<lossFuncType>(output, target, delta);
  layers_.backward(delta);
}

matrix_t Network::forwardProp(const matrix_t& input) {
  return layers_.forward(input);
}

void Network::updateWeights(Optimizer* optimizer) {
  layers_.updateWeights(optimizer);
}

LayerSequence* Network::getLayerSequence() {
  return &layers_;
}

matrix_t Network::predict(const matrix_t& input) {
  return forwardProp(input);
}

template<LossFunctionType lossFuncType, typename Optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
bool Network::fit(Optimizer &optimizer, const matrix_t& inputs, const std::vector<label_t>& classLabels,
  size_t batchSize, size_t numberOfEpochs, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate) {

  if (inputs.rows() != classLabels.size() || inputs.rows() < batchSize) {
    return false;
  }

  matrix_t labelsAsTensor;
  convertLabelsToMatrix(classLabels, labelsAsTensor, getOutDataSize(), getTargetValueRange());
  return fit<lossFuncType>(optimizer, inputs, labelsAsTensor, batchSize, numberOfEpochs, onBatchEnumerate, onEpochEnumerate);
}

template<LossFunctionType lossFuncType, typename Optimizer>
bool Network::fit(Optimizer &optimizer, const matrix_t& inputs, const matrix_t& desiredOutputs, size_t batchSize, int epoch) {
  return fit<lossFuncType>(optimizer, inputs, desiredOutputs, batchSize, epoch, [](){}, [](){});
}

template<LossFunctionType lossFuncType>
float Network::getLoss(const matrix_t& input, const matrix_t& target) {
  matrix_t predicted = predict(input);
  return LossCalculator::getLoss<lossFuncType>(predicted, target);
}

template<LossFunctionType lossFuncType>
float Network::getLoss(const matrix_t& input, const std::vector<label_t>& expectedLabels) {
  matrix_t target;  
  convertLabelsToMatrix(expectedLabels, target, getOutDataSize(), getTargetValueRange());
  return getLoss<lossFuncType>(input, target);
}

size_t Network::getOutDataSize() const {
  return layers_.getOutDataSize();
}

size_t Network::getInDataSize() const {
  return layers_.getInDataSize();
}

template<LossFunctionType lossFuncType, typename Optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
bool Network::fit(Optimizer& optimizer, const matrix_t& inputs, const matrix_t& targets,
  size_t batchSize, size_t numberOfEpochs, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate) {

  layers_.initialize();
  optimizer.reset();

  for (size_t epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
    for (size_t batchOffset = 0; batchOffset < inputs.rows(); batchOffset += batchSize) {
      size_t batchIndexLast;
      if (batchOffset + batchSize - 1 < inputs.rows()) {
        batchIndexLast = batchOffset + batchSize - 1;
      } else {
        batchIndexLast = inputs.rows() - 1;
      }
      fitBatch<lossFuncType>(optimizer, inputs, targets, batchOffset, batchIndexLast);
      onBatchEnumerate();
    }
    onEpochEnumerate();
  }
  return true;
}

template <LossFunctionType lossFuncType, typename Optimizer>
void Network::fitBatch(Optimizer &optimizer, const matrix_t& input,
  const matrix_t& target, size_t batchIndexFirst, size_t batchIndexLast) {

  size_t batchSize = batchIndexLast - batchIndexFirst + 1;
  matrix_t batchedInput(batchSize, input.cols());
  matrix_t batchedTarget(batchSize, target.cols());

  size_t batcPosIndex = 0;
  for (size_t sampleIndex = batchIndexFirst; sampleIndex <= batchIndexLast; sampleIndex++) {
    for (size_t colIndex = 0; colIndex < input.cols(); colIndex++) {
      batchedInput.at(batcPosIndex, colIndex) = input.at(sampleIndex, colIndex);
    }
    for (size_t colIndex = 0; colIndex < target.cols(); colIndex++) {
      batchedTarget.at(batcPosIndex, colIndex) = target.at(sampleIndex, colIndex);
    }
    batcPosIndex++;
  }

  backProp<lossFuncType>(forwardProp(batchedInput), batchedTarget);
  layers_.updateWeights(&optimizer); 
}

Result Network::test(const matrix_t& input, const std::vector<label_t>& expectedLabels) {

  Result testResult;
  std::vector<matrix_t> splittedInput;
  normalizeMatrix(input, splittedInput);
  for (size_t sampleIdx = 0; sampleIdx < splittedInput.size(); sampleIdx++) {
    matrix_t result = forwardProp(splittedInput[sampleIdx]);
    label_t predictedLabel = getIndexOfMaxElementInRow(result, 0);    
    testResult.addSingleTestSampleResult(predictedLabel == expectedLabels[sampleIdx]);
  }
  return testResult;
}

std::pair<float, float> Network::getTargetValueRange() const {
  return layers_.getLayer(layers_.getSize() - 1) -> getOutValueRange();
}

}