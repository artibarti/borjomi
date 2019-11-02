#pragma once

#include <string>
#include <utility>

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"
#include "borjomi/kernels/kernels.h"
#include "borjomi/network/layers/pooling_layer.h"

namespace borjomi {

class AveragePoolingLayer : public PoolingLayer {

 public:
  AveragePoolingLayer(size_t inWidth, size_t inHeight, size_t inChannels,
    size_t poolingSize, engine_t engine = engine_t::internal)
    : PoolingLayer(inWidth, inHeight, inChannels, poolingSize, engine) {}

  AveragePoolingLayer(AveragePoolingLayer&& other) : PoolingLayer(std::move(other)) {}

  std::string getLayerType() const override {
    return "average-pooling Layer";
  }

  void forwardPropagation() override {

    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");

    averagePoolForwardOp(getEngine(), inData, outData,
      getInputShape(), getOutputShape(), poolingSize());
  }

  void backPropagation() override {

    matrix_t& previousDelta = getEdgeGradient("incomingEdge");
    matrix_t& currentDelta = getEdgeGradient("outgoingEdge");

    averagePoolBackwardOp(getEngine(), previousDelta, currentDelta,
      getInputShape(), getOutputShape(), poolingSize());
  }
};

}