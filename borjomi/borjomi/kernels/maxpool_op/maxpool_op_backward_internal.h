#pragma once

#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void maxpoolInternalBackward(matrix_t& prevDelta, const matrix_t& currDelta,
  const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& maxIndices, size_t poolingSize) {

  for (size_t sampleIdx = 0; sampleIdx < prevDelta.rows(); sampleIdx++) {
    for (size_t maxIndiceIdx = 0; maxIndiceIdx < maxIndices.cols(); maxIndiceIdx++) {
      size_t maxIdx = maxIndices.at(sampleIdx, maxIndiceIdx);
      prevDelta.at(sampleIdx, maxIdx) = currDelta.at(sampleIdx, maxIndiceIdx);
    }
  }
}

}
}