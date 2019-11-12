#pragma once

#include <string>

#include "borjomi/network/activations/activation_layer.h"
#include "borjomi/network/layers/layer.h"

namespace borjomi {

class SoftmaxLayer : public ActivationLayer {

 public:
  using ActivationLayer::ActivationLayer;

  std::string getLayerType() const override {
    return "softmax-activation";
  }

  void forwardActivation(const matrix_t& x, matrix_t& y) override {
    for(size_t rowIdx = 0; rowIdx < x.rows(); rowIdx++) {
      float denominator(0);
      float alpha = x.at(rowIdx, 0);
      for (size_t colIndex = 1; colIndex < x.cols(); colIndex++) {
        if (x.at(rowIdx, colIndex) > alpha) {
          alpha = x.at(rowIdx, colIndex);
        }
      }
      for (size_t colIdx = 0; colIdx < x.cols(); colIdx++) {
        y.at(rowIdx, colIdx) = std::exp(x.at(rowIdx, colIdx) - alpha);
        denominator += y.at(rowIdx, colIdx);
      }
      for (size_t colIdx = 0; colIdx < x.cols(); colIdx++) {
        y.at(rowIdx, colIdx) /= denominator;
      }
    }
  }

  void backwardActivation(const matrix_t& x, const matrix_t& y, matrix_t& dx, const matrix_t& dy) override {

    for(size_t rowIdx = 0; rowIdx < x.rows(); rowIdx++) {
      for (size_t j = 0; j < x.cols(); j++) {
        std::vector<float> df(x.cols());
        // float* df = new float[x.cols()];
        for (size_t k = 0; k < x.cols(); k++) {
          df[k] = (k == j) ? y.at(rowIdx, j) * (float(1) - y.at(rowIdx, j)) : -y.at(rowIdx, k) * y.at(rowIdx, j);
        }
        float sum = 0;
        for (size_t k = 0; k < x.cols(); k++) {
          sum += (dy.at(rowIdx, k) * df[k]);
        }
        dx.at(rowIdx, j) = sum;
      }
    }
  }

  std::pair<float, float> scale() const override {
    return std::make_pair(float(0), float(1));
  }
};

}