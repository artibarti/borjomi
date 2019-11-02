#pragma once

#include <string>
#include <utility>
#include <vector>

#include "borjomi/network/layers/layer.h"
#include "borjomi/util/util.h"

namespace borjomi {

class ActivationLayer : public Layer {

 public:
  ActivationLayer(size_t dim = 0) : ActivationLayer(shape3d_t(1, dim, 1)) {}
  ActivationLayer(const shape3d_t& shape) : Layer(shape, shape) {}

  void forwardPropagation() override {
    matrix_t& inData = getEdgeData("incomingEdge");
    matrix_t& outData = getEdgeData("outgoingEdge");
    forwardActivation(inData, outData);
  }

  void backPropagation() override {
    matrix_t& dInData = getEdgeGradient("incomingEdge");
    const matrix_t& dOutData = getEdgeGradient("outgoingEdge");
    const matrix_t& inData = getEdgeData("incomingEdge");
    const matrix_t& outData = getEdgeData("outgoingEdge");
    backwardActivation(inData, outData, dInData, dOutData);
  }

  void setBatchSize(size_t batchSize = 1) override {
  
    shape2d_t inShape(batchSize, getInDataSize());
    shape2d_t outShape(batchSize, getOutDataSize());
    reshapeEdge("incomingEdge", inShape);
    reshapeEdge("outgoingEdge", outShape);
  }

  virtual void forwardActivation(const matrix_t& x, matrix_t &y) = 0;
  virtual void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) = 0;

  virtual std::pair<float, float> scale() const = 0;

 protected:
  std::map<std::string, float> params;
};

}