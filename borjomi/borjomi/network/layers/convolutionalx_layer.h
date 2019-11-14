#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"
#include "borjomi/kernels/kernels.h"
#include "borjomi/network/layers/trainable_layer.h"

namespace borjomi {

class ConvolutionalXLayer : public TrainableLayer {

 private:
  shape2d_t weightsShape_;
  shape3d_t weightShape_;

  size_t kernelSize_;

  shape3d_t reorganizedInShape_;

  matrix_t reorganizedInData_;
  matrix_t reorganizedPrevDelta_;

 public:
  ConvolutionalXLayer(size_t inWidth, size_t inHeight, size_t inChannels, size_t kernelSize,
    size_t outChannels, padding paddingType = padding::valid, bool hasBias = true, engine_t engine = engine_t::internal)
    : TrainableLayer(shape3d_t(inWidth, inHeight, inChannels),
      shape3d_t(inWidth, inHeight, outChannels), hasBias, engine) {

    kernelSize_ = kernelSize;
    weightShape_ = shape3d_t(kernelSize, inChannels, kernelSize);
    weightsShape_ = shape2d_t(outChannels, weightShape_.size());
    reorganizedInShape_ = shape3d_t(inHeight, inChannels, inWidth);
  }

  ConvolutionalXLayer(ConvolutionalXLayer&& other) 
    : TrainableLayer(std::move(other)) {
    kernelSize_ = other.kernelSize_;
    weightShape_ = other.weightShape_;
    weightsShape_ = other.weightsShape_;
    reorganizedPrevDelta_ = other.reorganizedPrevDelta_;
    reorganizedInData_ = other.reorganizedInData_;
    reorganizedInShape_ = other.reorganizedInShape_;
  }

  void initialize() override {

    size_t fanInSize = weightShape_.rows_ * weightShape_.cols_ * getInputShape().channels_;
    size_t fanOutSize = weightShape_.rows_ * weightShape_.cols_ * getOutputShape().channels_;

    if (getEdge("weight") -> shape() != weightsShape_) {
      reshapeEdge("weight", weightsShape_);
      WeightInitializer::initialize(WeightInitializerType::xavier,
        getEdgeData("weight"), fanInSize, fanOutSize);
    }

    shape2d_t biasShape = shape2d_t(1, getOutputShape().channels_);
    if (getEdge("bias") -> shape() != biasShape) {
      reshapeEdge("bias", biasShape);
      WeightInitializer::initialize(WeightInitializerType::constant,
        getEdgeData("bias"), fanInSize, fanOutSize);
    }
  }

  void setBatchSize(size_t batchSize = 1) override {
    TrainableLayer::setBatchSize(batchSize);
    reorganizedInData_.reshape(batchSize, getInputShape().size());
    reorganizedPrevDelta_.reshape(batchSize, getInputShape().size());
  }

  void forwardPropagation() override {

    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& bias = getEdgeData("bias");

    shuffleColAndChannelDimesions(inData, getInputShape(), reorganizedInData_);

    convxForwardOp(engine_t::internal, reorganizedInData_, weights,
      bias, outData, reorganizedInShape_, getOutputShape(), weightShape_);
  }

  void backPropagation() override {

    matrix_t& prevOut = getEdgeData("incomingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& dWeight = getEdgeGradient("weight");
    matrix_t& db = getEdgeGradient("bias");
    matrix_t& prevDelta = getEdgeGradient("incomingEdge");
    matrix_t& currDelta = getEdgeGradient("outgoingEdge");

    for (size_t idx = 0; idx < prevDelta.size(); idx++) {
      prevDelta.at(idx) = 0;
    }

    convxBackwardOp(getEngine(), prevOut, weights, dWeight, db,
      currDelta, prevDelta, weightShape_, getInputShape(), getOutputShape());
  }

  std::string getLayerType() const override {
    return "convolutional-layer-v2";
  }

 private:
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
};

}