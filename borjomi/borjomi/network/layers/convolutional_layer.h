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

class ConvolutionalLayer : public TrainableLayer {

 private:
  size_t padding_;
  shape3d_t weight_;
  shape3d_t inPadded_;

 public:
  ConvolutionalLayer(size_t inWidth, size_t inHeight, size_t inChannels, size_t kernelWidth, size_t kernelHeight,
    size_t outChannels, padding paddingType = padding::valid, bool hasBias = true, engine_t engine = engine_t::internal)
    : TrainableLayer(shape3d_t(inWidth, inHeight, inChannels), shape3d_t(inWidth, inHeight, outChannels), hasBias, engine) {

    shape3d_t inShape = shape3d_t(inWidth, inHeight, inChannels);
    shape3d_t outShape = shape3d_t(inWidth, inHeight, outChannels);

    padding_ = 2;
    setConvParams(inShape, outShape, kernelWidth, kernelHeight, paddingType);
  }

  ConvolutionalLayer(ConvolutionalLayer &&other)  : TrainableLayer(std::move(other)),
    params_(std::move(other.params_)), padOp_(std::move(other.padOp_)) {

    prevOutPadded_ = other.prevOutPadded_;
    prevDeltaPadded_ = other.prevDeltaPadded_;
    padding_ = other.padding_;
  }

  void initialize() override {

    size_t fanInSize = params_.weight.rows_ * params_.weight.cols_ * params_.in.channels_;
    size_t fanOutSize = params_.weight.rows_ * params_.weight.cols_ * params_.out.channels_;

    shape2d_t weightShape = shape2d_t(1, params_.weight.size());
    if (getEdge("weight") -> shape() != weightShape) {
      reshapeEdge("weight", weightShape);
      WeightInitializer::initialize(WeightInitializerType::xavier,
        getEdgeData("weight"), fanInSize, fanOutSize);
    }

    shape2d_t biasShape = shape2d_t(1, params_.out.channels_);
    if (getEdge("bias") -> shape() != biasShape) {
      reshapeEdge("bias", biasShape);
      WeightInitializer::initialize(WeightInitializerType::constant,
        getEdgeData("bias"), fanInSize, fanOutSize);
    }
  }

  void setBatchSize(size_t batchSize = 1) override {
    TrainableLayer::setBatchSize(batchSize);
    prevDeltaPadded_.reshape(batchSize, params_.in_padded.size());
  }

  void forwardPropagation() override {

    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& bias = getEdgeData("bias");

    for (size_t idx = 0; idx < outData.size(); idx++) {
      outData.at(idx) = float{0};
    }

    padOp_.addPadding(inData, prevOutPadded_, padding_);
    convForwardOp(getEngine(), *inDataPadded(&inData), weights, bias, outData, params_);
  }

  void backPropagation() override {

    matrix_t& prevOut = getEdgeData("incomingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& dWeight = getEdgeGradient("weight");
    matrix_t& db = getEdgeGradient("bias");
    matrix_t& prevDelta = getEdgeGradient("incomingEdge");
    matrix_t& currDelta = getEdgeGradient("outgoingEdge");

    for (size_t idx = 0; idx < prevDeltaPadded_.size(); idx++) {
      prevDeltaPadded_.at(idx) = float{0};
    }

    convBackwardOp(getEngine(), *inDataPadded(&prevOut), weights, dWeight, db, currDelta, prevDeltaPadded_, params_);
    padOp_.removePadding(prevDeltaPadded_, prevDelta, padding_);
  }

  std::string getLayerType() const override {
    return "convolutional-layer";
  }

 private:
  matrix_t prevOutPadded_;
  matrix_t prevDeltaPadded_;

  kernels::conv_params params_;
  kernels::Conv2dPadding padOp_;

  matrix_t* inDataPadded(matrix_t* in) {
    return (params_.pad_type == padding::valid) ? in : &prevOutPadded_;
  }

  void setConvParams(const shape3d_t& in, const shape3d_t& out,
    size_t windowWidth, size_t windowHeight, padding paddingType) {

    params_.in = in;
    params_.in_padded = shape3d_t(inLength(in.rows_, windowWidth, paddingType), inLength(in.cols_, windowHeight, paddingType), in.channels_);
    params_.out = out;
    params_.weight = shape3d_t(windowWidth, windowHeight, in.channels_ * out.channels_);
    params_.pad_type = paddingType;

    if (params_.pad_type == padding::same) {
      prevDeltaPadded_.reshape(1, params_.in_padded.size());
    }
    padOp_ = kernels::Conv2dPadding(params_);    
  }

  size_t inLength(size_t inLength, size_t windowSize, padding padType) const {
    return padType == padding::same ? (inLength + windowSize - 1) : inLength;
  }
};

}