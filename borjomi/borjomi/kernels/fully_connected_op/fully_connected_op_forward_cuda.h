#pragma once

#include "borjomi/types/types.h"
#include "borjomi/engine/engine.h"

namespace borjomi {
namespace kernels {

  void fullyConnectedForwardCuda(const matrix_t& inData, const matrix_t& weights,
    const matrix_t& bias, matrix_t& outData) {

    if (!bias.isEmpty()) {
      engine::internal::copy(0.0, false, bias.rows(), bias.cols(),
        bias.data(), outData.rows(), outData.cols(), outData.data());
    }
    engine::avx::multiply(1.0, false, inData.rows(), inData.cols(), inData.data(),
      true, weights.rows(), weights.cols(), weights.data(), 1.0, outData.data());
  }
}
}