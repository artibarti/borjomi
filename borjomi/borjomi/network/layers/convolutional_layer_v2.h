/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
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

class ConvolutionalLayerV2 : public TrainableLayer {

 private:
  shape3d_t weightShape_;
  size_t kernelSize_;

 public:
  ConvolutionalLayerV2(size_t inWidth, size_t inHeight, size_t inChannels, size_t kernelSize,
    size_t outChannels, padding paddingType = padding::valid, bool hasBias = true, engine_t engine = engine_t::internal)
    : TrainableLayer(shape3d_t(inWidth, inHeight, inChannels),
      shape3d_t(inWidth, inHeight, outChannels), hasBias, engine) {

    shape3d_t inShape = shape3d_t(inWidth, inHeight, inChannels);
    shape3d_t outShape = shape3d_t(inWidth, inHeight, outChannels);
    kernelSize_ = kernelSize;
    weightShape_ = shape3d_t(kernelSize, kernelSize, inChannels * outChannels);
  }

  ConvolutionalLayerV2(ConvolutionalLayerV2&& other) 
    : TrainableLayer(std::move(other)) {
    kernelSize_ = other.kernelSize_;
    weightShape_ = other.weightShape_;
  }

  void initialize() override {

    size_t fanInSize = weightShape_.rows_ * weightShape_.cols_ * getInputShape().channels_;
    size_t fanOutSize = weightShape_.rows_ * weightShape_.cols_ * getOutputShape().channels_;

    shape2d_t weightShape = shape2d_t(1, weightShape_.size());
    if (getEdge("weight") -> shape() != weightShape) {
      reshapeEdge("weight", weightShape);
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
  }

  void forwardPropagation() override {

    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& bias = getEdgeData("bias");

    matrix_t reorganizedInput(inData.shape());
    matrix_t reorganizedWeights(weights.shape());

    for (size_t idx = 0; idx < outData.size(); idx++) {
      outData.at(idx) = 0;
    }

    convv2ForwardOp(getEngine(), reorganizedInput, reorganizedWeights,
      bias, outData, getInputShape(), getOutputShape(), kernelSize_);
  }

  void backPropagation() override {

    // matrix_t& prevOut = getEdgeData("incomingEdge");
    // matrix_t& weights = getEdgeData("weight");
    // matrix_t& dWeight = getEdgeGradient("weight");
    // matrix_t& db = getEdgeGradient("bias");
    // matrix_t& prevDelta = getEdgeGradient("incomingEdge");
    // matrix_t& currDelta = getEdgeGradient("outgoingEdge");

    // for (size_t idx = 0; idx < prevDelta.size(); idx++) {
    //   prevDelta.at(idx) = 0;
    // }
  }

  std::string getLayerType() const override {
    return "convolutional-layer-v2";
  }

 private:

  void reorganizeInputForConvolutionalLayerForwardOp(const matrix_t& input,
    const shape3d_t& originalShape, matrix_t& reshaped) {

    shape3d_t newShape(originalShape.cols_, originalShape.channels_, originalShape.rows_);
    reshaped.reshape(input.rows(), newShape.size_);
    for (size_t sampleIdx = 0; sampleIdx < input.rows(); sampleIdx++) {
      for (size_t originalChannelIdx = 0; originalChannelIdx < originalShape.channels_; originalChannelIdx++) {
        for (size_t originalRowIdx = 0; originalRowIdx < originalShape.rows_; originalRowIdx++) {
          for (size_t originalColIdx = 0; originalColIdx < originalShape.cols_; originalColIdx++) {
            reshaped.at(sampleIdx, newShape.getIndex(originalColIdx, originalChannelIdx, originalRowIdx))
              = input.at(sampleIdx, originalShape.getIndex(originalRowIdx, originalColIdx, originalChannelIdx));
          }
        }
      }
    }
  }

  void reorganizeWeights(const matrix_t& weights, matrix_t& reorganizedWeights,
    size_t kernelSize, const shape3d_t& inShape, const shape3d_t& outShape) {

    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      size_t idx = 0;
      for (size_t rowIdx = 0; rowIdx < kernelSize; rowIdx++) {
        for (size_t colIdx = 0; colIdx < kernelSize; colIdx++) {
          for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {
            size_t originalIdx = (outChannelIdx * inShape.channels_ + inChannelIdx) * (kernelSize * kernelSize)
              + rowIdx * kernelSize + colIdx;
            size_t newIdx = (outChannelIdx * inShape.channels_) * (kernelSize * kernelSize) + idx;
            reorganizedWeights.at(0, newIdx) = weights.at(0, originalIdx);
            idx++;
          }
        }
      }
    }
  }
};

}