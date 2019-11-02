#pragma once

#include <algorithm>

#include "borjomi/types/types.h"
#include "borjomi/util/math/random.h"

namespace borjomi {

enum class WeightInitializerType {
  xavier, lecun, gaussian, constant, he
};

class WeightInitializer {

 public:
  static void initialize(WeightInitializerType initializerType, matrix_t& weight, size_t inSize, size_t outSize) {
    
    if (initializerType == WeightInitializerType::xavier) {
      xavier(weight, inSize, outSize);
    } else if (initializerType == WeightInitializerType::lecun) {
      lecun(weight, inSize, outSize);
    } else if (initializerType == WeightInitializerType::gaussian) {
      gaussian(weight, inSize, outSize);
    } else if (initializerType == WeightInitializerType::constant) {
      constant(weight, inSize, outSize);
    } else if (initializerType == WeightInitializerType::he) {
      he(weight, inSize, outSize);
    } else {
      throw BorjomiRuntimeException("WeightInitializer::initialize(): initializer type is unknown");
    }
  }

 private:
  static void xavier(matrix_t& weight, size_t inSize, size_t outSize) {
    float weightBase = std::sqrt(6.0 / (inSize + outSize));
    uniformRand(weight.data(), -weightBase, weightBase, weight.size());
  }

  static void lecun(matrix_t& weight, size_t inSize, size_t outSize) {
    float weightBase = 1.0 / std::sqrt(float(inSize));
    uniformRand(weight.data(), -weightBase, weightBase, weight.size());    
  }

  static void gaussian(matrix_t& weight, size_t inSize, size_t outSize) {
    gaussianRand(weight.data(), float{0}, 1.0, weight.size());    
  }

  static void constant(matrix_t& weight, size_t inSize, size_t outSize) {
    for (size_t idx = 0; idx < weight.size(); idx++) {
      weight.at(idx) = 0.0;
    }
  }

  static void he(matrix_t& weight, size_t inSize, size_t outSize) {
    float sigma = std::sqrt(1.0 / inSize);
    gaussianRand(weight.data(), float{0}, sigma, weight.size());
  }
};

}