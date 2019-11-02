#pragma once

#include <limits>
#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void averagePoolInternalForward(const matrix_t& inData, matrix_t& outData,
  const shape3d_t& inShape, const shape3d_t& outShape, size_t poolingSize) {
  
  for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
    for (size_t channelIdx = 0; channelIdx < outShape.channels_; channelIdx++) {
      for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
        for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
          float sum = 0;
          for (size_t inRowIdx = outRowIdx * poolingSize; inRowIdx < (outRowIdx + 1) * poolingSize; inRowIdx++) {
            for (size_t inColIdx = outColIdx * poolingSize; inColIdx < (outColIdx + 1) * poolingSize; inColIdx++) {
              sum += inData.at(sampleIdx, inShape.getIndex(inRowIdx, inColIdx, channelIdx));
            }
          }
          float average = sum / outShape.area_;
          outData.at(sampleIdx, outShape.getIndex(outRowIdx, outColIdx, channelIdx)) = average;
        }
      }
    }
  }
}

}
}
