#pragma once

#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void averagePoolInternalBackward(matrix_t& prevDelta, const matrix_t& currDelta,
  const shape3d_t& inShape, const shape3d_t& outShape, size_t poolingSize) {

  for (size_t sampleIdx = 0; sampleIdx < prevDelta.rows(); sampleIdx++) {
    for (size_t channelIdx = 0; channelIdx < inShape.channels_; channelIdx++) {
      for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
        for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
          float sharedAverage = currDelta.at(sampleIdx, outShape.getIndex(outRowIdx, outColIdx, channelIdx))
            / (poolingSize * poolingSize);
          for (size_t inRowIdx = outRowIdx * poolingSize; inRowIdx < (outRowIdx + 1) * poolingSize; inRowIdx++) {
            for (size_t inColIdx = outColIdx * poolingSize; inColIdx < (outColIdx + 1) * poolingSize; inColIdx++) {
              prevDelta.at(sampleIdx, inShape.getIndex(inRowIdx, inColIdx, channelIdx)) = sharedAverage;
            }
          }
        }
      }
    }
  }
}

}
}
