#pragma once

#include <cmath>

#include "borjomi/types/types.h"

namespace borjomi {

enum class LossFunctionType {
  Mse, CrossEntropy
};

class LossCalculator {

 public:
  template<LossFunctionType lossFunctionType>
  static float getLoss(const matrix_t& actual, const matrix_t& expected) {
    if (lossFunctionType == LossFunctionType::Mse) {
      return mse(actual, expected);
    } else if (lossFunctionType == LossFunctionType::CrossEntropy) {
      return crossEntropy(actual, expected);
    } else {
      throw BorjomiRuntimeException("LossCalculator::getLoss(): loss function type is not implemented");
    }
  }

  template<LossFunctionType lossFunctionType>
  static void getLossMatrix(const matrix_t& actual, const matrix_t& expected, matrix_t& result) {
    if (lossFunctionType == LossFunctionType::Mse) {
      mse(actual, expected, result);
    } else if (lossFunctionType == LossFunctionType::CrossEntropy) {
      crossEntropy(actual, expected, result);
    }
  }

 private:
  static float mse(const matrix_t& actual, const matrix_t& expected) {
    float distance = 0;
    for (size_t idx = 0; idx < actual.size(); idx++) {
      distance += (actual.at(idx) - expected.at(idx)) * (actual.at(idx) - expected.at(idx));
    }
    return distance / static_cast<float>(actual.size());
  }

  static void mse(const matrix_t& actual, const matrix_t& expected, matrix_t& result) {
    size_t sampleSize = actual.cols();
    float factor = float(2) / static_cast<float>(sampleSize);
    for (size_t idx = 0; idx < actual.size(); idx++) {
      result.at(idx) = factor * (actual.at(idx) - expected.at(idx));
    }
  }

  static float crossEntropy(const matrix_t& actual, const matrix_t& expected) {
    float distance = 0;
    for (size_t idx = 0; idx < actual.size(); idx++) {
      distance += -expected.at(idx) * std::log(actual.at(idx))
        - (float(1) - expected.at(idx)) * std::log(float(1) - actual.at(idx));
    }
    return distance;
  }

  static void crossEntropy(const matrix_t& actual, const matrix_t& expected, matrix_t& result) {
    for (size_t idx = 0; idx < actual.size(); idx++) {
      result.at(idx) = (actual.at(idx) - expected.at(idx)) /
        (actual.at(idx) * (float(1) - actual.at(idx)));
    }
  }
};

}