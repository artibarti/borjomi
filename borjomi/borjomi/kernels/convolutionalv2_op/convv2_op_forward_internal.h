#pragma once

#include <cmath>
#include <algorithm>

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convv2ForwardInternal(const matrix_t& inData, const matrix_t& W, const matrix_t& bias,
  matrix_t& outData, const shape3d_t& inShape, const shape3d_t& outShape, size_t kernelSize) {

  if (!bias.isEmpty()) {
    for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
      for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
        engine::internal::fill(&outData.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx)),
          outShape.area_, bias.at(0, outChannelIdx));
      }
    }
  }

  size_t padding = std::floor(kernelSize / 2);
  shape3d_t weightShape(kernelSize, inShape.cols_, kernelSize);

  for (size_t sampleIdx = 0; sampleIdx < inData.rows(); sampleIdx++) {
    for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
      for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
        for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {

          size_t inRowBegin = std::max(int(outRowIdx - padding), 0);
          size_t inRowEnd = std::min(int(outRowIdx + padding), int(inShape.rows_ - 1));
          size_t nRows = inRowEnd - inRowBegin + 1;

          size_t inChannelBegin = std::max(int(outColIdx - padding), 0);
          size_t inChannelEnd = std::min(int(outColIdx + padding), int(inShape.channels_ - 1));
          size_t nChannels = inChannelEnd - inChannelBegin + 1;

          size_t weightRowBegin = int(outRowIdx - padding) >= 0 ? 0 : kernelSize - nRows;
          size_t weightChannelBegin = int(outColIdx - padding) >= 0 ? 0 : kernelSize - nChannels;

          float sum = 0;

          for (size_t inChannelIdx = inChannelBegin, weightChannelIdx = weightChannelBegin;
            inChannelIdx <= inChannelEnd; inChannelIdx++, weightChannelIdx++) {

            for (size_t inRowIdx = inRowBegin, weightRowIdx = weightRowBegin;
              inRowIdx <= inRowEnd; inRowIdx++, weightRowIdx++) {

              for (size_t inColIdx = 0; inColIdx < inShape.cols_; inColIdx++) {
                float in = inData.at(sampleIdx, inShape.getIndex(inRowIdx, inColIdx, inChannelIdx));
                float w = W.at(outChannelIdx, weightShape.getIndex(weightRowIdx, inColIdx, weightChannelIdx));
                sum += w * in;
              }
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