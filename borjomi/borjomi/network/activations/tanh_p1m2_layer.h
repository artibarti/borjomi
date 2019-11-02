#pragma once

#include <string>

#include "borjomi/network/activations/activation_layer.h"
#include "borjomi/network/layers/layer.h"

namespace borjomi {

class Tanhp1m2Layer : public ActivationLayer {

 public:
  using ActivationLayer::ActivationLayer;

  std::string getLayerType() const override {
    return "tanh-scaled-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    float ep;
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      ep = std::exp(x.at(elementIdx));
      y.at(elementIdx) = ep / (ep + (float(1) / ep));
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      dx.at(elementIdx) = dy.at(elementIdx) * (2 * y.at(elementIdx) * (float(1) - y.at(elementIdx)));
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}