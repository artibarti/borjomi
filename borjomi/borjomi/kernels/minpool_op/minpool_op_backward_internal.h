#pragma once

#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void minpoolInternalBackward(matrix_t& prevDelta, const matrix_t& currDelta,
  const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& minIndices, size_t poolingSize) {

  for (size_t sampleIdx = 0; sampleIdx < prevDelta.rows(); sampleIdx++) {
    for (size_t minIndiceIdx = 0; minIndiceIdx < minIndices.cols(); minIndiceIdx++) {
      size_t maxIdx = minIndices.at(sampleIdx, minIndiceIdx);
      prevDelta.at(sampleIdx, maxIdx) = currDelta.at(sampleIdx, minIndiceIdx);
    }
  }
}

}
}