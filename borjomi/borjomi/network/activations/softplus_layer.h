#pragma once

#include <string>

#include "borjomi/network/activations/activation_layer.h"

namespace borjomi {

class SoftplusLayer : public ActivationLayer {

 public:
  SoftplusLayer(const float beta = 1.0, const float threshold = 20.0) : ActivationLayer() {
    params["beta"] = beta;
    params["threshold"] = threshold;
  }

  std::string getLayerType() const override {
    return "softplus-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      float betain = params["beta"] * x.at(elementIdx);
      y.at(elementIdx) = (betain > params["threshold"])
        ? x.at(elementIdx) : (1 / params["beta"]) * std::log1p(std::exp(betain));
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {
    for (size_t elementIdx = 0; elementIdx < x.size(); elementIdx++) {
      float betaout = params["beta"] * y.at(elementIdx);
      float exp_bo  = std::exp(betaout);
      dx.at(elementIdx) = (betaout > params["threshold"])
        ? dy.at(elementIdx) : dy.at(elementIdx) * (exp_bo - 1) / exp_bo;
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0.1), float(0.9));
  }
};

}