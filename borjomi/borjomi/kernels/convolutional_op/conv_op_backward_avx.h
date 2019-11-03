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

void accumulateBiasDelta(const shape3d_t& out, const float* currDelta, float* db) {

    auto area = out.area();
    size_t n8 = area / 64;
    size_t n4 = (area % 64) / 32;
    size_t n2 = (area % 32) / 16;
    size_t n1 = (area % 16) / 8;
    size_t remainder = area & 7;

    static const int32_t masks[] = {
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    __m256i mask = _mm256_loadu_si256((const __m256i *)(masks + 8 - remainder));

    for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
      
      size_t idx = out.getIndex(0, 0, outChannelIdx);
      
      const float *delta = &currDelta[idx];
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();
      __m256 sum4 = _mm256_setzero_ps();
      __m256 sum5 = _mm256_setzero_ps();
      __m256 sum6 = _mm256_setzero_ps();
      __m256 sum7 = _mm256_setzero_ps();
      
      for (size_t i = 0; i < n8; ++i) {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(delta + i * 64 + 0));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(delta + i * 64 + 8));
        sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(delta + i * 64 + 16));
        sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(delta + i * 64 + 24));
        sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(delta + i * 64 + 32));
        sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(delta + i * 64 + 40));
        sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(delta + i * 64 + 48));
        sum7 = _mm256_add_ps(sum7, _mm256_loadu_ps(delta + i * 64 + 56));
      }
      
      delta += n8 * 64;
      
      if (n4) {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(delta + 0));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(delta + 8));
        sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(delta + 16));
        sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(delta + 24));
        delta += 32;
      }
      if (n2) {
        sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(delta + 0));
        sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(delta + 8));
        delta += 16;
      }
      if (n1) {
        sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(delta));
        delta += 8;
      }
      sum0 = _mm256_add_ps(sum0, sum1);
      sum2 = _mm256_add_ps(sum2, sum3);
      sum4 = _mm256_add_ps(sum4, sum5);
      sum6 = _mm256_add_ps(sum6, sum7);
      sum1 = _mm256_maskload_ps(delta, mask);
      sum0 = _mm256_add_ps(sum0, sum2);
      sum4 = _mm256_add_ps(sum4, sum6);
      sum0 = _mm256_add_ps(sum0, sum4);
      sum0 = _mm256_add_ps(sum0, sum1);
      db[outChannelIdx] += _mm_cvtss_f32(hsum256_ps(sum0));
    }
}

void accumulateWeightDelta(const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape, const float* prevOut,
  const float* currDelta, float* dW) {
  
  auto &in = inShape;
  auto &out = outShape;
  auto &in_padded = inPaddedShape;
  auto &weight = weightShape;

  const size_t nblocks = out.cols_ >> 3;
  static const int32_t masks[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const size_t remainder = out.rows_ & 7;
  __m256i mask = _mm256_loadu_si256((const __m256i *)(masks + 8 - remainder));
  size_t prevo_delta = in_padded.cols_ * 1;
  __m256 sum0, sum1, sum2, sum3, sum4;

    if (nblocks == 1 && remainder != 0) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *delta = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          size_t widx        = weight.getIndex(0, 0, in.channels_ * outChannelIdx + inputChannelIdx);
          float *pdw         = &dW[widx];
          // weight.rows_
          for (size_t weightRowIdx = 0; weightRowIdx < 5; weightRowIdx++) {
            size_t prevOutIdx = in_padded.getIndex(weightRowIdx, 0, inputChannelIdx);
            const float *pa     = &prevOut[prevOutIdx];
            const float *pb     = delta;
            // y = 0
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out.rows_; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              a0        = _mm256_maskload_ps(pa + 010, mask);
              a1        = _mm256_maskload_ps(pa + 011, mask);
              a2        = _mm256_maskload_ps(pa + 012, mask);
              a3        = _mm256_maskload_ps(pa + 013, mask);
              a4        = _mm256_maskload_ps(pa + 014, mask);
              b         = _mm256_maskload_ps(pb + 010, mask);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out.cols_;
            }
            _mm_storeu_ps(pdw + weightRowIdx * 5, _mm_add_ps(_mm_loadu_ps(pdw + weightRowIdx * 5), hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(pdw + weightRowIdx * 5 + 4, _mm_add_ss(_mm_load_ss(pdw + weightRowIdx * 5 + 4), hsum256_ps(sum4)));
          }
        }
      }
    } else if (nblocks > 1 && remainder != 0) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *delta = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          size_t widx = weight.getIndex(0, 0, in.channels_ * outChannelIdx + inputChannelIdx);
          float *pdw = &dW[widx];
          for (size_t weightRowIdx = 0; weightRowIdx < 5; weightRowIdx++) {
            size_t prevOutIdx = in_padded.getIndex(weightRowIdx, 0, inputChannelIdx);
            const float *pa     = &prevOut[prevOutIdx];
            const float *pb     = delta;
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out.rows_;
                 ++y, pa += prevo_delta, pb += out.cols_) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              a0 = _mm256_loadu_ps(pa + 010);
              a1 = _mm256_loadu_ps(pa + 011);
              a2 = _mm256_loadu_ps(pa + 012);
              a3 = _mm256_loadu_ps(pa + 013);
              a4 = _mm256_loadu_ps(pa + 014);
              b = _mm256_loadu_ps(pb + 010);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              for (size_t i = 2; i < nblocks; ++i) {
                a0 = _mm256_loadu_ps(pa + 8 * i + 0);
                a1 = _mm256_loadu_ps(pa + 8 * i + 1);
                a2 = _mm256_loadu_ps(pa + 8 * i + 2);
                a3 = _mm256_loadu_ps(pa + 8 * i + 3);
                a4 = _mm256_loadu_ps(pa + 8 * i + 4);
                b = _mm256_loadu_ps(pb + 8 * i);
                sum0 = madd256_ps(a0, b, sum0);
                sum1 = madd256_ps(a1, b, sum1);
                sum2 = madd256_ps(a2, b, sum2);
                sum3 = madd256_ps(a3, b, sum3);
                sum4 = madd256_ps(a4, b, sum4);
              }
              a0 = _mm256_maskload_ps(pa + 8 * nblocks + 0, mask);
              a1 = _mm256_maskload_ps(pa + 8 * nblocks + 1, mask);
              a2 = _mm256_maskload_ps(pa + 8 * nblocks + 2, mask);
              a3 = _mm256_maskload_ps(pa + 8 * nblocks + 3, mask);
              a4 = _mm256_maskload_ps(pa + 8 * nblocks + 4, mask);
              b = _mm256_maskload_ps(pb + 8 * nblocks, mask);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
            }
            _mm_storeu_ps(pdw + weightRowIdx * 5, _mm_add_ps(_mm_loadu_ps(pdw + weightRowIdx * 5), hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(pdw + weightRowIdx * 5 + 4, _mm_add_ss(_mm_load_ss(pdw + weightRowIdx * 5 + 4), hsum256_ps(sum4)));
          }
        }
      }
    } else if (nblocks == 0) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *delta = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          size_t widx = weight.getIndex(0, 0, in.channels_ * outChannelIdx + inputChannelIdx);
          float *pdw = &dW[widx];
          // weight.rows_
          for (size_t weightRowIdx = 0; weightRowIdx < 5; weightRowIdx++) {
            size_t prevOutIdx = in_padded.getIndex(weightRowIdx, 0, inputChannelIdx);
            const float *pa = &prevOut[prevOutIdx];
            const float *pb = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out.rows_; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_maskload_ps(pa + 0, mask);
              __m256 a1 = _mm256_maskload_ps(pa + 1, mask);
              __m256 a2 = _mm256_maskload_ps(pa + 2, mask);
              __m256 a3 = _mm256_maskload_ps(pa + 3, mask);
              __m256 a4 = _mm256_maskload_ps(pa + 4, mask);
              __m256 b  = _mm256_maskload_ps(pb, mask);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out.cols_;
            }
            _mm_storeu_ps(pdw + weightRowIdx * 5, _mm_add_ps(_mm_loadu_ps(pdw + weightRowIdx * 5), hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(pdw + weightRowIdx * 5 + 4, _mm_add_ss(_mm_load_ss(pdw + weightRowIdx * 5 + 4), hsum256_ps(sum4)));
          }
        }
      }
    } else if (nblocks == 1) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *delta = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          size_t widx = weight.getIndex(0, 0, in.channels_ * outChannelIdx + inputChannelIdx);
          float *pdw = &dW[widx];
          // weight.rows_
          for (size_t weightRowIdx = 0; weightRowIdx < 5; weightRowIdx++) {
            size_t prevOutIdx = in_padded.getIndex(weightRowIdx, 0, inputChannelIdx);
            const float *pa     = &prevOut[prevOutIdx];
            const float *pb     = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out.rows_; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out.cols_;
            }
            _mm_storeu_ps(pdw + weightRowIdx * 5, _mm_add_ps(_mm_loadu_ps(pdw + weightRowIdx * 5), hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(pdw + weightRowIdx * 5 + 4, _mm_add_ss(_mm_load_ss(pdw + weightRowIdx * 5 + 4), hsum256_ps(sum4)));
          }
        }
      }
    } else {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *delta = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          size_t widx        = weight.getIndex(0, 0, in.channels_ * outChannelIdx + inputChannelIdx);
          float *pdw         = &dW[widx];
          // weight.rows_
          for (size_t weightRowIdx = 0; weightRowIdx < 5; weightRowIdx++) {
            size_t prevOutIdx = in_padded.getIndex(weightRowIdx, 0, inputChannelIdx);
            const float *pa     = &prevOut[prevOutIdx];
            const float *pb     = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out.rows_; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b = _mm256_loadu_ps(pb);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              a0 = _mm256_loadu_ps(pa + 010);
              a1 = _mm256_loadu_ps(pa + 011);
              a2 = _mm256_loadu_ps(pa + 012);
              a3 = _mm256_loadu_ps(pa + 013);
              a4 = _mm256_loadu_ps(pa + 014);
              b = _mm256_loadu_ps(pb + 010);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
              for (size_t i = 2; i < nblocks; ++i) {
                a0   = _mm256_loadu_ps(pa + 8 * i + 0);
                a1   = _mm256_loadu_ps(pa + 8 * i + 1);
                a2   = _mm256_loadu_ps(pa + 8 * i + 2);
                a3   = _mm256_loadu_ps(pa + 8 * i + 3);
                a4   = _mm256_loadu_ps(pa + 8 * i + 4);
                b    = _mm256_loadu_ps(pb + 8 * i);
                sum0 = madd256_ps(a0, b, sum0);
                sum1 = madd256_ps(a1, b, sum1);
                sum2 = madd256_ps(a2, b, sum2);
                sum3 = madd256_ps(a3, b, sum3);
                sum4 = madd256_ps(a4, b, sum4);
              }
              pa += prevo_delta;
              pb += out.cols_;
            }
            _mm_storeu_ps(pdw + weightRowIdx * 5,  _mm_add_ps(_mm_loadu_ps(pdw + weightRowIdx * 5), hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(pdw + weightRowIdx * 5 + 4, _mm_add_ss(_mm_load_ss(pdw + weightRowIdx * 5 + 4), hsum256_ps(sum4)));
          }
        }
      }
    }
}

void avx_conv2d_5x5_back_kernel_one(const shape3d_t& inShape, const shape3d_t& inPaddedShape,
  const shape3d_t& outShape, const shape3d_t& weightShape, const float* prevOut,
  const float* W, float* dW, float* db, const float* currDelta, float* prevDelta) {

  auto &in = inShape;
  auto &out = outShape;
  auto &in_padded = inPaddedShape;
  const size_t in_padded_area = in_padded.area();
  float *pdelta_dst_org = &(prevDelta[0]);

  static const __m256i imask  = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
  static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));

  if (out.cols_ >= 4) {
    const size_t nblocks = out.cols_ / 4;
    if (out.cols_ % 4) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++, pdelta_dst_org += in_padded_area) {

        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {

          const float *pw = &W[25 * (in.channels_ * outChannelIdx + inputChannelIdx)];
          const float *pdelta_src = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          float *pdelta_dst = pdelta_dst_org;

          __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw + 0), mask);
          __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw + 5), mask);
          __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw + 10), mask);
          __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw + 15), mask);
          __m256 w4a = _mm256_maskload_ps(pw + 20, imask);
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

          for (size_t y = 0; y < out.rows_; ++y, pdelta_src += out.cols_, pdelta_dst += in_padded.rows_) {
            
            float *delta_dst0 = pdelta_dst;
            float *delta_dst1 = &pdelta_dst[in_padded.cols_ * 1];
            float *delta_dst2 = &pdelta_dst[in_padded.cols_ * 2];
            float *delta_dst3 = &pdelta_dst[in_padded.cols_ * 3];
            float *delta_dst4 = &pdelta_dst[in_padded.cols_ * 4];
            
            for (size_t n = 0; n < nblocks; ++n) {
              __m256 delta_src = _mm256_broadcast_ps((const __m128 *)(pdelta_src + n * 4));
              __m256 dst0 = _mm256_loadu_ps(delta_dst0 + 4 * n);
              __m256 dst1 = _mm256_loadu_ps(delta_dst1 + 4 * n);
              __m256 dst2 = _mm256_loadu_ps(delta_dst2 + 4 * n);
              __m256 dst3 = _mm256_loadu_ps(delta_dst3 + 4 * n);
              __m256 dst4 = _mm256_loadu_ps(delta_dst4 + 4 * n);
              __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
              __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
              __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
              __m256 delta_src3 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
              dst0 = madd256_ps(w0a, delta_src0, dst0);
              dst1 = madd256_ps(w1a, delta_src0, dst1);
              dst2 = madd256_ps(w2a, delta_src0, dst2);
              dst3 = madd256_ps(w3a, delta_src0, dst3);
              dst4 = madd256_ps(w4a, delta_src0, dst4);
              dst0 = madd256_ps(w0b, delta_src1, dst0);
              dst1 = madd256_ps(w1b, delta_src1, dst1);
              dst2 = madd256_ps(w2b, delta_src1, dst2);
              dst3 = madd256_ps(w3b, delta_src1, dst3);
              dst4 = madd256_ps(w4b, delta_src1, dst4);
              dst0 = madd256_ps(w0c, delta_src2, dst0);
              dst1 = madd256_ps(w1c, delta_src2, dst1);
              dst2 = madd256_ps(w2c, delta_src2, dst2);
              dst3 = madd256_ps(w3c, delta_src2, dst3);
              dst4 = madd256_ps(w4c, delta_src2, dst4);
              dst0 = madd256_ps(w0d, delta_src3, dst0);
              _mm256_storeu_ps(delta_dst0 + 4 * n, dst0);
              dst1 = madd256_ps(w1d, delta_src3, dst1);
              _mm256_storeu_ps(delta_dst1 + 4 * n, dst1);
              dst2 = madd256_ps(w2d, delta_src3, dst2);
              _mm256_storeu_ps(delta_dst2 + 4 * n, dst2);
              dst3 = madd256_ps(w3d, delta_src3, dst3);
              _mm256_storeu_ps(delta_dst3 + 4 * n, dst3);
              dst4 = madd256_ps(w4d, delta_src3, dst4);
              _mm256_storeu_ps(delta_dst4 + 4 * n, dst4);
            }

            for (size_t x = nblocks * 4; x < out.cols_; ++x) {
              __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
              __m256 dst0 = _mm256_loadu_ps(delta_dst0 + x);
              __m256 dst1 = _mm256_loadu_ps(delta_dst1 + x);
              __m256 dst2 = _mm256_loadu_ps(delta_dst2 + x);
              __m256 dst3 = _mm256_loadu_ps(delta_dst3 + x);
              __m256 dst4 = _mm256_maskload_ps(delta_dst4 + x, imask);
              dst0 = madd256_ps(w0a, delta_src, dst0);
              dst1 = madd256_ps(w1a, delta_src, dst1);
              dst2 = madd256_ps(w2a, delta_src, dst2);
              dst3 = madd256_ps(w3a, delta_src, dst3);
              dst4 = madd256_ps(w4a, delta_src, dst4);
              _mm256_maskstore_ps(delta_dst0 + x, imask, dst0);
              _mm256_maskstore_ps(delta_dst1 + x, imask, dst1);
              _mm256_maskstore_ps(delta_dst2 + x, imask, dst2);
              _mm256_maskstore_ps(delta_dst3 + x, imask, dst3);
              _mm256_maskstore_ps(delta_dst4 + x, imask, dst4);
            }
          }
        }
      }
    } else {
      for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++, pdelta_dst_org += in_padded_area) {
        for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {
          const float *pw = &W[25 * (in.channels_ * outChannelIdx + inputChannelIdx)];
          const float *pdelta_src = &currDelta[out.getIndex(0, 0, outChannelIdx)];
          float *pdelta_dst = pdelta_dst_org;
          __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw + 0), mask);
          __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw + 5), mask);
          __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw + 10), mask);
          __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw + 15), mask);
          __m256 w4a = _mm256_maskload_ps(pw + 20, imask);
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
          size_t y   = 0;
          do {
            float *delta_dst0 = pdelta_dst;
            float *delta_dst1 = &pdelta_dst[in_padded.cols_ * 1];
            float *delta_dst2 = &pdelta_dst[in_padded.cols_ * 2];
            float *delta_dst3 = &pdelta_dst[in_padded.cols_ * 3];
            float *delta_dst4 = &pdelta_dst[in_padded.cols_ * 4];
            size_t n          = 0;
            do {
              __m256 delta_src = _mm256_broadcast_ps((const __m128 *)(pdelta_src + n * 4));
              __m256 dst0 = _mm256_loadu_ps(delta_dst0 + 4 * n);
              __m256 dst1 = _mm256_loadu_ps(delta_dst1 + 4 * n);
              __m256 dst2 = _mm256_loadu_ps(delta_dst2 + 4 * n);
              __m256 dst3 = _mm256_loadu_ps(delta_dst3 + 4 * n);
              __m256 dst4 = _mm256_loadu_ps(delta_dst4 + 4 * n);
              __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
              __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
              __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
              __m256 delta_src3 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
              dst0 = madd256_ps(w0a, delta_src0, dst0);
              dst1 = madd256_ps(w1a, delta_src0, dst1);
              dst2 = madd256_ps(w2a, delta_src0, dst2);
              dst3 = madd256_ps(w3a, delta_src0, dst3);
              dst4 = madd256_ps(w4a, delta_src0, dst4);
              dst0 = madd256_ps(w0b, delta_src1, dst0);
              dst1 = madd256_ps(w1b, delta_src1, dst1);
              dst2 = madd256_ps(w2b, delta_src1, dst2);
              dst3 = madd256_ps(w3b, delta_src1, dst3);
              dst4 = madd256_ps(w4b, delta_src1, dst4);
              dst0 = madd256_ps(w0c, delta_src2, dst0);
              dst1 = madd256_ps(w1c, delta_src2, dst1);
              dst2 = madd256_ps(w2c, delta_src2, dst2);
              dst3 = madd256_ps(w3c, delta_src2, dst3);
              dst4 = madd256_ps(w4c, delta_src2, dst4);
              dst0 = madd256_ps(w0d, delta_src3, dst0);
              _mm256_storeu_ps(delta_dst0 + 4 * n, dst0);
              dst1 = madd256_ps(w1d, delta_src3, dst1);
              _mm256_storeu_ps(delta_dst1 + 4 * n, dst1);
              dst2 = madd256_ps(w2d, delta_src3, dst2);
              _mm256_storeu_ps(delta_dst2 + 4 * n, dst2);
              dst3 = madd256_ps(w3d, delta_src3, dst3);
              _mm256_storeu_ps(delta_dst3 + 4 * n, dst3);
              dst4 = madd256_ps(w4d, delta_src3, dst4);
              _mm256_storeu_ps(delta_dst4 + 4 * n, dst4);
              ++n;
            } while (n < nblocks);
            ++y;
            pdelta_src += out.cols_;
            pdelta_dst += in_padded.rows_;
          } while (y < out.rows_);
        }
      }
    }
  } else if (out.rows_ == 1 && out.cols_ == 1) {
    for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++, pdelta_dst_org += in_padded_area) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m128 sum3 = _mm_setzero_ps();

      size_t widx  = 25 * inputChannelIdx;
      size_t wstep = 25 * in.channels_;
      __m256 delta_src;

      for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++, widx += wstep) {
        delta_src = _mm256_broadcast_ss(&currDelta[outChannelIdx]);
        const float *pw = (const float *)&W[widx];
        __m256 w0 = _mm256_loadu_ps(pw + 0);
        __m256 w1 = _mm256_loadu_ps(pw + 8);
        __m256 w2 = _mm256_loadu_ps(pw + 16);
        __m128 w3 = _mm_load_ss(pw + 24);
        sum0 = madd256_ps(w0, delta_src, sum0);
        sum1 = madd256_ps(w1, delta_src, sum1);
        sum2 = madd256_ps(w2, delta_src, sum2);
        sum3 = madd128_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
      }

      float *delta_dst0 = pdelta_dst_org;
      float *delta_dst1 = &pdelta_dst_org[in_padded.cols_ * 1];
      float *delta_dst2 = &pdelta_dst_org[in_padded.cols_ * 2];
      float *delta_dst3 = &pdelta_dst_org[in_padded.cols_ * 3];
      float *delta_dst4 = &pdelta_dst_org[in_padded.cols_ * 4];
      
      __m256 dst0 = _mm256_loadu_ps(delta_dst0);
      __m256 dst1 = _mm256_loadu_ps(delta_dst1);
      __m256 dst2 = _mm256_loadu_ps(delta_dst2);
      __m256 dst3 = _mm256_loadu_ps(delta_dst3);
      __m256 dst4 = _mm256_maskload_ps(delta_dst4, imask);

      __m256 new_sum0 = _mm256_blend_ps(_mm256_setzero_ps(), sum0, 0x1F);
      __m256 new_sum1 = _mm256_blend_ps(_mm256_setzero_ps(), _mm256_or_ps(rightShift<20>(sum0), leftShift<12>(sum1)), 0x1F);
      __m256 new_sum2 = _mm256_blend_ps(_mm256_setzero_ps(), rightShift<8>(sum1), 0x1F);
      __m256 new_sum3 = _mm256_blend_ps(_mm256_setzero_ps(), _mm256_or_ps(rightShift<28>(sum1), leftShift<4>(sum2)), 0x1F);
      __m256 new_sum4 = _mm256_blend_ps(_mm256_setzero_ps(), _mm256_set_m128(sum3, _mm256_extractf128_ps(sum2, 1)), 0x1F);

      dst0 = _mm256_add_ps(dst0, new_sum0);
      dst1 = _mm256_add_ps(dst1, new_sum1);
      dst2 = _mm256_add_ps(dst2, new_sum2);
      dst3 = _mm256_add_ps(dst3, new_sum3);
      dst4 = _mm256_add_ps(dst4, new_sum4);

      _mm256_maskstore_ps(delta_dst0, imask, dst0);
      _mm256_maskstore_ps(delta_dst1, imask, dst1);
      _mm256_maskstore_ps(delta_dst2, imask, dst2);
      _mm256_maskstore_ps(delta_dst3, imask, dst3);
      _mm256_maskstore_ps(delta_dst4, imask, dst4);
    }
  } else {
    for (size_t inputChannelIdx = 0; inputChannelIdx < in.channels_; inputChannelIdx++, pdelta_dst_org += in_padded_area) {
      for (size_t outChannelIdx = 0; outChannelIdx < out.channels_; outChannelIdx++) {

        const float *pw = &W[25 * (in.channels_ * outChannelIdx + inputChannelIdx)];
        const float *pdelta_src = &currDelta[out.getIndex(0, 0, outChannelIdx)];
        float *pdelta_dst = pdelta_dst_org;
        __m256 w0a = _mm256_maskload_ps(pw + 0, imask);
        __m256 w1a = _mm256_maskload_ps(pw + 5, imask);
        __m256 w2a = _mm256_maskload_ps(pw + 10, imask);
        __m256 w3a = _mm256_maskload_ps(pw + 15, imask);
        __m256 w4a = _mm256_maskload_ps(pw + 20, imask);
        for (size_t y = 0; y < out.rows_; ++y, pdelta_src += out.cols_, pdelta_dst += in_padded.rows_) {
          float *delta_dst0 = pdelta_dst;
          float *delta_dst1 = &pdelta_dst[in_padded.cols_ * 1];
          float *delta_dst2 = &pdelta_dst[in_padded.cols_ * 2];
          float *delta_dst3 = &pdelta_dst[in_padded.cols_ * 3];
          float *delta_dst4 = &pdelta_dst[in_padded.cols_ * 4];
          for (size_t x = 0; x < out.cols_; ++x) {
            __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
            __m256 dst0 = _mm256_loadu_ps(delta_dst0);
            __m256 dst1 = _mm256_loadu_ps(delta_dst1);
            __m256 dst2 = _mm256_loadu_ps(delta_dst2);
            __m256 dst3 = _mm256_loadu_ps(delta_dst3);
            __m256 dst4 = _mm256_maskload_ps(delta_dst4, imask);
            dst0 = madd256_ps(w0a, delta_src, dst0);
            dst1 = madd256_ps(w1a, delta_src, dst1);
            dst2 = madd256_ps(w2a, delta_src, dst2);
            dst3 = madd256_ps(w3a, delta_src, dst3);
            dst4 = madd256_ps(w4a, delta_src, dst4);
            _mm256_storeu_ps(delta_dst0, dst0);
            _mm256_storeu_ps(delta_dst1, dst1);
            _mm256_storeu_ps(delta_dst2, dst2);
            _mm256_storeu_ps(delta_dst3, dst3);
            _mm256_maskstore_ps(delta_dst4, imask, dst4);
            delta_dst0++;
            delta_dst1++;
            delta_dst2++;
            delta_dst3++;
            delta_dst4++;
          }
        }
      }
    }
  }
}

void convBackwardAvx(const matrix_t& prevOut, const matrix_t& W, matrix_t& dW,
  matrix_t& db, const matrix_t& currDelta, matrix_t& prevDelta, const shape3d_t& inShape,
  const shape3d_t& inPaddedShape, const shape3d_t& outShape, const shape3d_t& weightShape) {

  engine::threads::parallelized2DLoop(prevOut.rows(), 1, 1, 1, [&](size_t sampleIdx, size_t toIgnore) {
    avx_conv2d_5x5_back_kernel_one(inShape, inPaddedShape, outShape, weightShape, &prevOut.at(sampleIdx, 0), &W.at(0), &dW.at(0),
      &db.at(0), &currDelta.at(sampleIdx, 0), &prevDelta.at(sampleIdx, 0));
    accumulateWeightDelta(inShape, inPaddedShape, outShape, weightShape, &prevOut.at(sampleIdx, 0), &currDelta.at(sampleIdx, 0), &dW.at(0));      
    if (!db.isEmpty()) {
      accumulateBiasDelta(outShape, &currDelta.at(sampleIdx, 0), &db.at(0));
    }
  });
}

}
}