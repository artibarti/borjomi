#pragma once

#include "borjomi/kernels/params/conv_params.h"
#include "borjomi/engine/engine.h"

namespace borjomi {
namespace kernels {

void convForwardThreads(const matrix_t& inData, const matrix_t& W, const matrix_t& bias,
  matrix_t& outData, const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape) {

  if (!bias.isEmpty()) {
    engine::threads::parallelized2DLoop(inData.rows(), outShape.channels_, 1, 1, [&](size_t sampleIdx, size_t outChannelIdx) {
      engine::threads::fill(&outData.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx)), outShape.area_, bias.at(0, outChannelIdx));
    });
  }
    
  engine::threads::parallelized2DLoop(inData.rows(),
    outShape.channels_, 1, 1, [&](size_t sampleIdx, size_t outChannelIdx) {      

    size_t outBaseIdx = outShape.getIndex(0, 0, outChannelIdx);
    float* output = &outData.at(sampleIdx, outBaseIdx);
    size_t outIdx = 0;

    for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
      for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {
        float sum = 0;
        for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {
          size_t weightBaseIdx = weightShape.getIndex(0, 0, outChannelIdx * inShape.channels_ + inChannelIdx);
          const float* weightBase = &W.at(weightBaseIdx);
          size_t kernelIdx = 0;
          for (size_t kernelRowIdx = 0; kernelRowIdx < 5; kernelRowIdx++) {
            for (size_t kernelColIdx = 0; kernelColIdx < 5; kernelColIdx++) {
              size_t inIdx = inPaddedShape.getIndex(kernelRowIdx + outRowIdx, kernelColIdx + outColIdx, inChannelIdx);
              sum += weightBase[kernelIdx] * inData.at(sampleIdx, inIdx);
              kernelIdx++;
            }
          }
        }
        output[outIdx] += sum;
        outIdx++;
      }
    }
  });
}

}
}