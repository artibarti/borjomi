#pragma once

#include "borjomi/types/types.h"
#include "borjomi/engine/engine.h"

namespace borjomi {
namespace kernels {

  void fullyConnectedBackwardCuda(const matrix_t& prevOut, const matrix_t& weights,
    matrix_t& dWeights, matrix_t& dBias, const matrix_t& currDelta, matrix_t& prevDelta) {

    engine::avx::multiply(1.0f, false, currDelta.rows(), currDelta.cols(), currDelta.data(),
      false, weights.rows(), weights.cols(), weights.data(), 1.0f, prevDelta.data());

    engine::avx::multiply(1.0f, true, currDelta.rows(), currDelta.cols(), currDelta.data(),
      false, prevOut.rows(), prevOut.cols(), prevOut.data(), 1.0f, dWeights.data());

    if (!dBias.isEmpty()) {
      matrix_t identity(currDelta.rows(), 1, float{1});
      engine::avx::multiply(1.0, true, currDelta.rows(), currDelta.cols(), currDelta.data(),
        false, identity.rows(), identity.cols(), identity.data(), 1.0, dBias.data());
    }
  }
}
}