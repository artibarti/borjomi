#pragma once

#include <limits>
#include "borjomi/types/types.h"
#include "borjomi/engine/engine.h"

namespace borjomi {
namespace kernels {

void maxpoolForwardThreads(const matrix_t& inData, matrix_t& outData, const shape3d_t& inShape,
  const shape3d_t& outShape, matrix_i& maxIndices, size_t poolingSize) {
  
  engine::threads::parallelized2DLoop(inData.rows(), inShape.channels_, 1, 1, [&](size_t sampleIdx, size_t channelIdx) {
    const float* in = &inData.at(sampleIdx, inShape.getIndex(0,0,channelIdx));
    float* out = &outData.at(sampleIdx, outShape.getIndex(0,0,channelIdx));

    for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
      for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
        size_t maxIdx = 0;
        float maxValue = std::numeric_limits<float>::lowest();
        for (size_t inRowIdx = outRowIdx * poolingSize; inRowIdx < (outRowIdx + 1) * poolingSize; inRowIdx++) {
          for (size_t inColIdx = outColIdx * poolingSize; inColIdx < (outColIdx + 1) * poolingSize; inColIdx++) {
            float inVal = in[inRowIdx * inShape.cols_ + inColIdx];
            if (inVal > maxValue) {
              maxValue = inVal;
              maxIdx = inShape.getIndex(inRowIdx, inColIdx, channelIdx);
            }
          }
        }
        size_t destIdx = outShape.getIndex(outRowIdx, outColIdx, channelIdx);
        out[outRowIdx * outShape.cols_ + outColIdx] = maxValue;
        maxIndices.at(sampleIdx, destIdx) = maxIdx;
      }
    }
  });
}

}
}
