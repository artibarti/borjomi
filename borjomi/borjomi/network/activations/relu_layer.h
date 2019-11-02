#pragma once

#include <algorithm>
#include <string>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class ReluLayer : public ActivationLayer {
 
 public:
  using ActivationLayer::ActivationLayer;

  std::string getLayerType() const override {
    return "relu-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {    
    engine::threads::parallelized2DLoop(x.size(), 1, 1, 1, [&](size_t elementIdx, size_t justLeaveMeAlone) {
        y.at(elementIdx) = std::max(float(0), x.at(elementIdx));
    });
  }

  void backwardActivation(const matrix_t &x, const matrix_t &y, matrix_t &dx, const matrix_t &dy) override {
    engine::threads::parallelized2DLoop(x.size(), 1, 1, 1, [&](size_t elementIdx, size_t justLeaveMeAlone) {
      dx.at(elementIdx) = dy.at(elementIdx) * (y.at(elementIdx) > float(0) ? float(1) : float(0));
    });
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}