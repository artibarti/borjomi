#pragma once

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convBackwardThreads(const matrix_t& prevOut, const matrix_t& W, matrix_t& dW, matrix_t& db,
  const matrix_t& currDelta, matrix_t& prevDelta, const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape) {

  engine::threads::parallelized2DLoop(prevOut.rows(), inShape.channels_, 1, 1, [&](size_t sampleIdx, size_t inChannelIdx) {
    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {

      const float *pw = &W.at(0, weightShape.getIndex(0, 0, inShape.channels_ * outChannelIdx + inChannelIdx));
      const float *pdelta_src = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));
      float *pdelta_dst = &prevDelta.at(sampleIdx, inPaddedShape.getIndex(0, 0, inChannelIdx));

      for (size_t outputRowIdx = 0; outputRowIdx < outShape.cols_; outputRowIdx++) {
        for (size_t outputColIdx = 0; outputColIdx < outShape.rows_; outputColIdx++) {

          const float *ppw = pw;
          const float ppdelta_src = pdelta_src[outputRowIdx * outShape.rows_ + outputColIdx];
          float *ppdelta_dst = pdelta_dst + outputRowIdx * inPaddedShape.rows_ + outputColIdx;

          for (size_t weightRowIdx = 0; weightRowIdx < weightShape.cols_; weightRowIdx++) {
            for (size_t weightColIdx = 0; weightColIdx < weightShape.rows_; weightColIdx++) {
              ppdelta_dst[weightRowIdx * inPaddedShape.rows_ + weightColIdx] += *ppw++ * ppdelta_src;
            }
          }
        }
      }
    }
  });

  engine::threads::parallelized2DLoop(prevOut.rows(), inShape.channels_, 1, 1, [&](size_t sampleIdx, size_t inChannelIdx) {
    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      for (size_t weightRowIdx = 0; weightRowIdx < weightShape.cols_; weightRowIdx++) {
        for (size_t weightColIdx = 0; weightColIdx < weightShape.rows_; weightColIdx++) {

          float dst{0};
          const float *prevo = &prevOut.at(sampleIdx, inPaddedShape.getIndex(weightRowIdx, weightColIdx, inChannelIdx));
          const float *delta = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));

          for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
            float dot = 0;
            for (size_t idx = 0; idx < outShape.rows_; idx++) {
              dot += (prevo + outColIdx * inPaddedShape.rows_)[idx] * (delta + outColIdx * outShape.rows_)[idx];
            }
            dst += dot;
          }
          dW.at(0, weightShape.getIndex(weightRowIdx, weightColIdx, inShape.channels_ * outChannelIdx + inChannelIdx)) += dst;
        }
      }
    }
  });

  if (!db.isEmpty()) {
    engine::threads::parallelized2DLoop(prevOut.rows(),
      outShape.channels_, 1, 1, [&](size_t sampleIdx, size_t outChannelIdx) {
        const float *currDeltaChannelPtr = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));
        const float *currDeltaNextChannelPtr = currDeltaChannelPtr + outShape.area();
        db.at(0, outChannelIdx) += std::accumulate(currDeltaChannelPtr, currDeltaNextChannelPtr, float{0});
    });
  }
}

}
}