#pragma once

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convForwardInternal(const matrix_t& inData, const matrix_t& W,
  const matrix_t& bias, matrix_t& outData, const shape3d_t& inShape,
  const shape3d_t& inPaddedShape, const shape3d_t& outShape, const shape3d_t& weightShape) {

  if (!bias.isEmpty()) {
    for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
      for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
        engine::internal::fill(&outData.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx)),
          outShape.area_, bias.at(0, outChannelIdx));
      }
    }
  }
  for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
        for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
          for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {
            float sum = 0;
            for (size_t weightRowIdx = 0; weightRowIdx < weightShape.rows_; weightRowIdx++) {
              for (size_t weightColIdx = 0; weightColIdx < weightShape.cols_; weightColIdx++) {                
                size_t inIdx = inPaddedShape.getIndex(outRowIdx + weightColIdx, outColIdx + weightRowIdx, inChannelIdx);
                size_t weightIdx = weightShape.getIndex(weightRowIdx, weightColIdx, inShape.channels_ * outChannelIdx + inChannelIdx);
                sum += W.at(0, weightIdx) * inData.at(sampleIdx, inIdx);
              }
            }
            outData.at(sampleIdx, outShape.getIndex(outRowIdx, outColIdx, outChannelIdx)) += sum;
          }
        }
      }
    }
  }
}

}
}