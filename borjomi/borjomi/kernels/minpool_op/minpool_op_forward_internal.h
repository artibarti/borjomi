#pragma once

#include <limits>
#include "borjomi/types/types.h"

namespace borjomi {
namespace kernels {

void minpoolInternalForward(const matrix_t& inData, matrix_t& outData, const shape3d_t& inShape,
  const shape3d_t& outShape, matrix_i& minIndices, size_t poolingSize) {
  
  for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
    for (size_t channelIdx = 0; channelIdx < outShape.channels_; channelIdx++) {
      for (size_t rowGroupIdx = 0; rowGroupIdx < outShape.rows_; rowGroupIdx++) {
        for (size_t colGroupIdx = 0; colGroupIdx < outShape.cols_; colGroupIdx++) {
          size_t minIdx = 0;
          float minValue = std::numeric_limits<float>::max();
          for (size_t rowIdx = rowGroupIdx * poolingSize; rowIdx < (rowGroupIdx + 1) * poolingSize; rowIdx++) {
            for (size_t colIdx = colGroupIdx * poolingSize; colIdx < (colGroupIdx + 1) * poolingSize; colIdx++) {
              if (inData.at(sampleIdx, inShape.getIndex(rowIdx, colIdx, channelIdx)) < minValue) {
                minValue = inData.at(sampleIdx, inShape.getIndex(rowIdx, colIdx, channelIdx));
                minIdx = inShape.getIndex(rowIdx, colIdx, channelIdx);
              }
            }
          }
          size_t destIdx = outShape.getIndex(rowGroupIdx, colGroupIdx, channelIdx);
          outData.at(sampleIdx, destIdx) = minValue;
          minIndices.at(sampleIdx, destIdx) = minIdx;
        }
      }
    }
  }
}

}
}
