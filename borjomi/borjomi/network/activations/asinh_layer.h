#pragma once

#include <string>
#include <utility>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class AsinhLayer : public ActivationLayer {

 public:
  using ActivationLayer::ActivationLayer;

  std::string getLayerType() const override {
    return "asinh-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (unsigned elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      y.at(elementIdx) = std::asinh(x.at(elementIdx));
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (unsigned elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      dx.at(elementIdx) = dy.at(elementIdx) / std::cosh(y.at(elementIdx));
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(-0.8), float(0.8));
  }

};

}