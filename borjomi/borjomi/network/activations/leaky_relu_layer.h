#pragma once

#include <string>
#include <utility>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class LeakyReluLayer : public ActivationLayer {

 public:
  explicit LeakyReluLayer(unsigned dim, const float epsilon = 0.01) : LeakyReluLayer(shape3d_t(dim, 1, 1), epsilon) {}
  explicit LeakyReluLayer(const shape3d_t &shape, const float epsilon = 0.01) : ActivationLayer(shape) {
    params["epsilon"] = epsilon;
  }

  std::string getLayerType() const override {
    return "leaky-relu-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      y.at(elementIdx) = x.at(elementIdx) > float(0) ? x.at(elementIdx)
        : params["epsilon"] * x.at(elementIdx);
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      dx.at(elementIdx) = dy.at(elementIdx) * (y.at(elementIdx) > float(0) ? float(1) : params["epsilon"]);
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}