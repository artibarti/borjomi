#pragma once

#include <string>

#include "borjomi/network/activations/activation_layer.h"
#include "borjomi/network/layers/layer.h"

namespace borjomi {

class TanhLayer : public ActivationLayer {
 
 public:
  using ActivationLayer::ActivationLayer;

  std::string getLayerType() const override {
    return "tanh-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (size_t rowIdx = 0; rowIdx < x.shape().rows(); rowIdx++) {
      for(size_t colIdx = 0; colIdx < x.shape().cols(); colIdx++) {
        y.at(rowIdx, colIdx) = std::tanh(x.at(rowIdx, colIdx));
      }
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y,
    matrix_t& dx, const matrix_t& dy) override {

    for(size_t colIdx = 0; colIdx < x.shape().cols(); colIdx++) {
      for (size_t rowIdx = 0; rowIdx < x.shape().rows(); rowIdx++) {
        dx.at(rowIdx, colIdx) = dy.at(rowIdx, colIdx)
          * (float(1) - y.at(rowIdx, colIdx) * y.at(rowIdx, colIdx));
      }
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(-0.8), float(0.8));
  }
};

}