#pragma once

#include <cmath>
#include <algorithm>

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convv2ForwardInternal(const matrix_t& inData, const matrix_t& W, const matrix_t& bias,
  matrix_t& outData, const shape3d_t& inShape, const shape3d_t& outShape, size_t kernelSize) {

  size_t padding = kernelSize = std::ceil(kernelSize);
  for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
    for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
      for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {

        size_t inRowBegin = std::max(outRowIdx - padding, size_t{0});
        size_t inRowEnd = std::min(outRowIdx + padding, inShape.rows_ - size_t{1});
        size_t inColBegin = std::max(outColIdx - padding, size_t{0});
        size_t inColEnd = std::min(outColIdx + padding, inShape.cols_ - size_t{1});

      }
    }
  }
}

}
}