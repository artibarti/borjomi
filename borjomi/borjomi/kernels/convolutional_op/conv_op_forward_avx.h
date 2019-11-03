/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <immintrin.h>
#include <vector>

#include "borjomi/engine/engine.h"

namespace borjomi {
namespace kernels {

void convForwardAvx(const matrix_t& inData, const matrix_t& W,
  const matrix_t& bias, matrix_t& outData, const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape) {

  if (!bias.isEmpty()) {
    engine::threads::parallelized2DLoop(inData.rows(), outShape.channels_, 1, 1, [&](size_t sampleIdx, size_t outChannelIdx) {
      engine::avx::fill(&outData.at(sampleIdx, outShape.getIndex(0, 0, outChannelIdx)), outShape.area_, bias.at(0, outChannelIdx));
    });
  }

  engine::threads::parallelized2DLoop(inData.rows(), 1, 1, 1, [&](size_t sampleIdx, size_t toIgnore) {
    auto &out = outShape;
    auto &in_padded = inPaddedShape;

    const float* inSample = &inData.at(sampleIdx, 0);
    float* outSample = &outData.at(sampleIdx, 0);

    static const __m256i imask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);

    const size_t nblocks = out.rows_ / 4;
    const size_t outArea = out.area_;

    for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {      
      float* outChannel = &outSample[outChannelIdx * outArea];
      for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {

        const float *weightChannel = &W.at(25 * (inShape.channels_ * outChannelIdx + inChannelIdx));
        const float *inChannel = &inSample[in_padded.getIndex(0, 0, inChannelIdx)];

        __m256 w0a = _mm256_maskload_ps(weightChannel + 0, imask);
        __m256 w1a = _mm256_maskload_ps(weightChannel + 5, imask);
        __m256 w2a = _mm256_maskload_ps(weightChannel + 10, imask);
        __m256 w3a = _mm256_maskload_ps(weightChannel + 15, imask);
        __m256 w4a = _mm256_maskload_ps(weightChannel + 20, imask);

        __m256 w0b = leftShift<4>(w0a);
        __m256 w1b = leftShift<4>(w1a);
        __m256 w2b = leftShift<4>(w2a);
        __m256 w3b = leftShift<4>(w3a);
        __m256 w4b = leftShift<4>(w4a);

        __m256 w0c = leftShift<8>(w0a);
        __m256 w1c = leftShift<8>(w1a);
        __m256 w2c = leftShift<8>(w2a);
        __m256 w3c = leftShift<8>(w3a);
        __m256 w4c = leftShift<8>(w4a);

        __m256 w0d = leftShift<12>(w0a);
        __m256 w1d = leftShift<12>(w1a);
        __m256 w2d = leftShift<12>(w2a);
        __m256 w3d = leftShift<12>(w3a);
        __m256 w4d = leftShift<12>(w4a);

        float *ppa = outChannel;
          if (nblocks) {
            for (size_t outRowIdx = 0; outRowIdx < out.rows_; outRowIdx++, ppa += out.cols_) {
              const float *pi0 = (inChannel + outRowIdx * in_padded.cols_);
              const float *pi1 = pi0 + 1 * in_padded.cols_;
              const float *pi2 = pi0 + 2 * in_padded.cols_;
              const float *pi3 = pi0 + 3 * in_padded.cols_;
              const float *pi4 = pi0 + 4 * in_padded.cols_;
              __m256 dst0, dst1, dst2, dst3;
              __m256 i0 = _mm256_loadu_ps(pi0);
              __m256 i1 = _mm256_loadu_ps(pi1);
              __m256 i2 = _mm256_loadu_ps(pi2);
              __m256 i3 = _mm256_loadu_ps(pi3);
              __m256 i4 = _mm256_loadu_ps(pi4);
              dst0 = _mm256_mul_ps(w0a, i0);
              dst1 = _mm256_mul_ps(w0b, i0);
              dst2 = _mm256_mul_ps(w0c, i0);
              dst3 = _mm256_mul_ps(w0d, i0);
              dst0 = madd256_ps(w1a, i1, dst0);
              dst1 = madd256_ps(w1b, i1, dst1);
              dst2 = madd256_ps(w1c, i1, dst2);
              dst3 = madd256_ps(w1d, i1, dst3);
              dst0 = madd256_ps(w2a, i2, dst0);
              dst1 = madd256_ps(w2b, i2, dst1);
              dst2 = madd256_ps(w2c, i2, dst2);
              dst3 = madd256_ps(w2d, i2, dst3);
              dst0 = madd256_ps(w3a, i3, dst0);
              dst1 = madd256_ps(w3b, i3, dst1);
              dst2 = madd256_ps(w3c, i3, dst2);
              dst3 = madd256_ps(w3d, i3, dst3);
              dst0 = madd256_ps(w4a, i4, dst0);
              dst1 = madd256_ps(w4b, i4, dst1);
              dst2 = madd256_ps(w4c, i4, dst2);
              dst3 = madd256_ps(w4d, i4, dst3);
              __m128 sum = _mm_loadu_ps(ppa);
              __m128 hsum0123 = hsum4x256_ps(dst0, dst1, dst2, dst3);
              sum = _mm_add_ps(sum, hsum0123);
              _mm_storeu_ps(ppa, sum);
              for (size_t i = 1; i < nblocks; ++i) {
                i0 = _mm256_loadu_ps(pi0 + i * 4);
                i1 = _mm256_loadu_ps(pi1 + i * 4);
                i2 = _mm256_loadu_ps(pi2 + i * 4);
                i3 = _mm256_loadu_ps(pi3 + i * 4);
                i4 = _mm256_loadu_ps(pi4 + i * 4);
                dst0 = _mm256_mul_ps(w0a, i0);
                dst1 = _mm256_mul_ps(w0b, i0);
                dst2 = _mm256_mul_ps(w0c, i0);
                dst3 = _mm256_mul_ps(w0d, i0);
                dst0 = madd256_ps(w1a, i1, dst0);
                dst1 = madd256_ps(w1b, i1, dst1);
                dst2 = madd256_ps(w1c, i1, dst2);
                dst3 = madd256_ps(w1d, i1, dst3);
                dst0 = madd256_ps(w2a, i2, dst0);
                dst1 = madd256_ps(w2b, i2, dst1);
                dst2 = madd256_ps(w2c, i2, dst2);
                dst3 = madd256_ps(w2d, i2, dst3);
                dst0 = madd256_ps(w3a, i3, dst0);
                dst1 = madd256_ps(w3b, i3, dst1);
                dst2 = madd256_ps(w3c, i3, dst2);
                dst3 = madd256_ps(w3d, i3, dst3);
                dst0 = madd256_ps(w4a, i4, dst0);
                dst1 = madd256_ps(w4b, i4, dst1);
                dst2 = madd256_ps(w4c, i4, dst2);
                dst3 = madd256_ps(w4d, i4, dst3);
                sum  = _mm_loadu_ps(ppa + i * 4);
                hsum0123 = hsum4x256_ps(dst0, dst1, dst2, dst3);
                sum = _mm_add_ps(sum, hsum0123);
                _mm_storeu_ps(ppa + i * 4, sum);
              }
              for (size_t x = nblocks * 4; x < out.rows_; ++x) {
                sum = _mm_load_ss(&ppa[x]);
                i0 = _mm256_loadu_ps(pi0 + x);
                i1 = _mm256_loadu_ps(pi1 + x);
                i2 = _mm256_loadu_ps(pi2 + x);
                i3 = _mm256_loadu_ps(pi3 + x);
                i4 = _mm256_maskload_ps(pi4 + x, imask);
                __m256 sum0 = _mm256_mul_ps(w0a, i0);
                __m256 sum1 = _mm256_mul_ps(w1a, i1);
                sum0 = madd256_ps(w2a, i2, sum0);
                sum1 = madd256_ps(w3a, i3, sum1);
                sum0 = madd256_ps(w4a, i4, sum0);
                sum0 = _mm256_add_ps(sum0, sum1);
                _mm_store_ss(&ppa[x], _mm_add_ss(sum, hsum256_ps(sum0)));
              }
            }
          } else {
            for (size_t outputRowIdx = 0; outputRowIdx < out.cols_; outputRowIdx++, ppa += out.rows_) {
              const float *pi0 = (inChannel + outputRowIdx * in_padded.cols_);
              const float *pi1 = pi0 + 1 * in_padded.rows_;
              const float *pi2 = pi0 + 2 * in_padded.rows_;
              const float *pi3 = pi0 + 3 * in_padded.rows_;
              const float *pi4 = pi0 + 4 * in_padded.rows_;
              for (size_t outputRowIdx = 0; outputRowIdx < out.rows_; outputRowIdx++) {
                __m128 sum = _mm_load_ss(&ppa[outputRowIdx]);
                __m256 i0 = _mm256_loadu_ps(pi0 + outputRowIdx);
                __m256 i1 = _mm256_loadu_ps(pi1 + outputRowIdx);
                __m256 i2 = _mm256_loadu_ps(pi2 + outputRowIdx);
                __m256 i3 = _mm256_loadu_ps(pi3 + outputRowIdx);
                __m256 i4 = _mm256_maskload_ps(pi4 + outputRowIdx, imask);
                __m256 sum0 = _mm256_mul_ps(w0a, i0);
                __m256 sum1 = _mm256_mul_ps(w1a, i1);
                sum0 = madd256_ps(w2a, i2, sum0);
                sum1 = madd256_ps(w3a, i3, sum1);
                sum0 = madd256_ps(w4a, i4, sum0);
                sum0 = _mm256_add_ps(sum0, sum1);
                _mm_store_ss(&ppa[outputRowIdx], _mm_add_ss(sum, hsum256_ps(sum0)));
              }
            }
          }
        }
      }
    });
  }
}
}