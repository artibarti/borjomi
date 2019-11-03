#pragma once

#include <limits>
#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void maxpoolForwardInternal(const matrix_t& inData, matrix_t& outData, const shape3d_t& inShape,
  const shape3d_t& outShape, matrix_i& maxIndices, size_t poolingSize) {
  
  for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
    for (size_t channelIdx = 0; channelIdx < outShape.channels_; channelIdx++) {
      for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
        for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
          size_t maxIdx = 0;
          float maxValue = std::numeric_limits<float>::lowest();
          for (size_t inRowIdx = outRowIdx * poolingSize; inRowIdx < (outRowIdx + 1) * poolingSize; inRowIdx++) {
            for (size_t inColIdx = outColIdx * poolingSize; inColIdx < (outColIdx + 1) * poolingSize; inColIdx++) {
              if (inData.at(sampleIdx, inShape.getIndex(inRowIdx, inColIdx, channelIdx)) > maxValue) {
                maxValue = inData.at(sampleIdx, inShape.getIndex(inRowIdx, inColIdx, channelIdx));
                maxIdx = inShape.getIndex(inRowIdx, inColIdx, channelIdx);
              }
            }
          }
          size_t destIdx = outShape.getIndex(outRowIdx, outColIdx, channelIdx);
          outData.at(sampleIdx, destIdx) = maxValue;
          maxIndices.at(sampleIdx, destIdx) = maxIdx;
        }
      }
    }
  }
}

}
}
