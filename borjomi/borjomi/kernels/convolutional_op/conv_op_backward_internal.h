#pragma once

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convBackwardInternal(const matrix_t& prevOut, const matrix_t& W, matrix_t& dW, matrix_t& db,
  const matrix_t& currDelta, matrix_t& prevDelta, const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape) {

  for (size_t sampleIdx = 0; sampleIdx < prevOut.rows(); sampleIdx++) {
    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {

        const float *pw = &W.at(0, weightShape.getIndex(0, 0, inShape.channels_ * outChannelIdx + inChannelIdx));
        const float *pdelta_src = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));
        float *pdelta_dst = &prevDelta.at(sampleIdx, inPaddedShape.getIndex(0, 0, inChannelIdx));

        for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
          for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {

            const float *ppw = pw;
            const float ppdelta_src = pdelta_src[outRowIdx * outShape.cols_ + outColIdx];
            float *ppdelta_dst = pdelta_dst + outRowIdx * inPaddedShape.rows_ + outColIdx;

            for (size_t weightRowIdx = 0; weightRowIdx < weightShape.rows_; weightRowIdx++) {
              for (size_t weightColIdx = 0; weightColIdx < weightShape.cols_; weightColIdx++) {
                ppdelta_dst[weightRowIdx * inPaddedShape.rows_ + weightColIdx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {
        for (size_t weightRowIdx = 0; weightRowIdx < weightShape.rows_; weightRowIdx++) {
          for (size_t weightColIdx = 0; weightColIdx < weightShape.cols_; weightColIdx++) {
            float dst = 0;

            const float *prevo = &prevOut.at(sampleIdx, inPaddedShape.getIndex(weightRowIdx, weightColIdx, inChannelIdx));
            const float *delta = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));

            for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
              float dot = 0;
              for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
                dot += (prevo + outRowIdx * inPaddedShape.cols_)[outColIdx]
                  * (delta + outRowIdx * outShape.cols_)[outColIdx];
              }
              dst += dot;
            }
            dW.at(0, weightShape.getIndex(weightRowIdx, weightColIdx, inShape.channels_ * outChannelIdx + inChannelIdx)) += dst;
          }
        }
      }
    }

    if (!db.isEmpty()) {
      for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
        const float *currDeltaChannelPtr = &currDelta.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx));
        const float *currDeltaNextChannelPtr = currDeltaChannelPtr + outShape.area();
        db.at(0, outChannelIdx) += std::accumulate(currDeltaChannelPtr, currDeltaNextChannelPtr, float{0});
      }
    }
  }
}

}
}