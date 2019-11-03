#pragma once

#include <string>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class SeluLayer : public ActivationLayer {

 public:
  SeluLayer(const float lambda = 1.05070, const float alpha  = 1.67326) : ActivationLayer() {
    params["lambda"] = lambda;
    params["alpha"] = alpha;
  }

  std::string getLayerType() const override {
    return "selu-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      y.at(elementIdx) = params["lambda"] * (x.at(elementIdx) > float(0)
        ? x.at(elementIdx) : params["alpha"] * (std::exp(x.at(elementIdx)) - float(1)));
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      dx.at(elementIdx) = dy.at(elementIdx) * params["lambda"]
        * (x.at(elementIdx) > float(0) ? 1.0 : params["alpha"] * std::exp(x.at(elementIdx)));
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}