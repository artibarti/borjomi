#pragma once

#include <string>
#include <utility>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class EluLayer : public ActivationLayer {

 public:
  EluLayer(const float alpha = 1.0) : ActivationLayer() {
    params["alpha"] = alpha;
  }

  std::string getLayerType() const override {
    return "elu-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      y.at(elementIdx) = x.at(elementIdx) < float(0)
        ? (params["alpha"] * (std::exp(x.at(elementIdx)) - float(1))) : x.at(elementIdx);
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      dx.at(elementIdx) = dy.at(elementIdx) * (y.at(elementIdx) > float(0)
        ? float(1) : (params["alpha"] + y.at(elementIdx)));
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}