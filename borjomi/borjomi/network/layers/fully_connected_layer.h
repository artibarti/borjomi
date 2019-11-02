#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"
#include "borjomi/kernels/kernels.h"
#include "borjomi/network/layers/layer.h"

namespace borjomi {

class FullyConnectedLayer : public TrainableLayer {

 public:
  FullyConnectedLayer(size_t inDim, size_t outDim,
    bool hasBias = true, engine_t engine = engine_t::internal)
    : TrainableLayer(inDim, outDim, hasBias, engine) {}

  FullyConnectedLayer(FullyConnectedLayer&& other)
    : TrainableLayer(std::move(other)) {}

  void forwardPropagation() override {

    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");
    matrix_t& weights = getEdgeData("weight");

    if (hasBias()) {
      matrix_t& bias = getEdgeData("bias");
      fullyConnectedForwardOp(getEngine(), inData, weights, bias, outData);
    } else {
      fullyConnectedForwardOp(getEngine(), inData, weights, matrix_t(), outData);
    }

    /*
    if (hasBias()) {
      copy(getEngine(), 0.0, false, getEdgeData("bias"), outData);
    }
    multiply(getEngine(), 1.0, false, inData, true, weights, 1.0, outData);
    */
  }

  void backPropagation() override {

    matrix_t& prevOut = getEdgeData("incomingEdge");
    matrix_t& prevDelta = getEdgeGradient("incomingEdge");
    matrix_t& currDelta = getEdgeGradient("outgoingEdge");
    matrix_t& weights = getEdgeData("weight");
    matrix_t& dWeights = getEdgeGradient("weight");

    if (hasBias()) {
      matrix_t& dBias = getEdgeGradient("bias");
      fullyConnectedBackwardOp(getEngine(), prevOut, weights, dWeights, dBias, currDelta, prevDelta);
    } else {
      matrix_t empty;
      fullyConnectedBackwardOp(getEngine(), prevOut, weights, dWeights, empty, currDelta, prevDelta);
    }

    /*
    multiply(getEngine(), 1.0f, false, currDelta, false, weights, 1.0f, prevDelta);
    multiply(getEngine(), 1.0f, true, currDelta, false, prevOut, 1.0f, dWeights);

    if (hasBias()) {
      matrix_t identity(currDelta.rows(), 1, float{1});
      multiply(getEngine(), 1.0, true, currDelta, false, identity, 1.0, getEdgeGradient("bias"));
    }
    */
  }

  void initialize() override {  

    shape2d_t weightShape(getOutDataSize(), getInDataSize());
    if (getEdge("weight") -> shape() != weightShape) {
      reshapeEdge("weight", weightShape);
      WeightInitializer::initialize(WeightInitializerType::xavier,
        getEdgeData("weight"), getInDataSize(), getOutDataSize());
    }

    if (hasBias()) {
      shape2d_t biasShape(1, getOutDataSize());
      if (getEdge("bias") -> shape() != biasShape) {
        reshapeEdge("bias", biasShape);
        WeightInitializer::initialize(WeightInitializerType::constant,
          getEdgeData("bias"), getInDataSize(), getOutDataSize());        
      }
    }
  }

  std::string getLayerType() const override {
    return "fully connected layer";
  }
};

}