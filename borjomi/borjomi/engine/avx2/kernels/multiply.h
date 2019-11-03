#pragma once

#include "borjomi/types/types.h"
#include "borjomi/engine/internal/kernels/kernels.h"

#include <immintrin.h>
#include "borjomi/engine/avx2/utils/utils.h"
#include "borjomi/engine/avx2/utils/utils2.h"

namespace borjomi {
 namespace engine {
  namespace avx {

  void multiply(float alpha, bool transLeft, size_t rowsInLeft, size_t colsInLeft, const float* left,
    bool transRight, size_t rowsInRight, size_t colsInRight, const float* right, float beta, float* result) {

   if (!transLeft && !transRight) {
      
      size_t destRowBlockNumb = rowsInLeft / 8;
      // size_t destRowRemains = rowsInLeft % 7;
      size_t destColBlockNumb = colsInRight / 8;
      size_t destColRemains = colsInRight & 7;
      size_t leftColBlockNumb = colsInLeft / 8;
      size_t leftColRemains = colsInLeft & 7;

      for (size_t destRowIdx = 0; destRowIdx < destRowBlockNumb * 8; destRowIdx += 8) {
        for (size_t destColIdx = 0; destColIdx < destColBlockNumb * 8; destColIdx += 8) {
  
          __m256 multiplier = _mm256_set1_ps(beta);
          __m256 dest[8] = {
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 0) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 1) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 2) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 3) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 4) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 5) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 6) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 7) * colsInRight + destColIdx])),
          };

          for (size_t leftColIdx = 0; leftColIdx < leftColBlockNumb * 8; leftColIdx += 8) {
            __m256 right_[8] = {
              _mm256_loadu_ps(&right[(leftColIdx + 0) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 1) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 2) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 3) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 4) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 5) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 6) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 7) * colsInRight + destColIdx]),
            };

            for (size_t leftRowIdx = destRowIdx; leftRowIdx < destRowIdx + 8; leftRowIdx++) {
              
              __m256 left0 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 0]);
              __m256 left1 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 1]);
              __m256 left2 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 2]);
              __m256 left3 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 3]);
              __m256 left4 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 4]);
              __m256 left5 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 5]);
              __m256 left6 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 6]);
              __m256 left7 = _mm256_set1_ps(left[leftRowIdx * colsInLeft + leftColIdx + 7]);

              size_t destIdx = leftRowIdx - destRowIdx;
              dest[destIdx] = _mm256_fmadd_ps(left0, right_[0], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left1, right_[1], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left2, right_[2], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left3, right_[3], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left4, right_[4], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left5, right_[5], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left6, right_[6], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left7, right_[7], dest[destIdx]);
            }
          }

          __m256 right_[leftColRemains];
          for (size_t leftColRemainIdx = leftColBlockNumb * 8; leftColRemainIdx < colsInLeft; leftColRemainIdx++) {
            right_[leftColRemainIdx - leftColBlockNumb * 8] = _mm256_loadu_ps(&right[leftColRemainIdx * colsInRight + destColIdx]);
          }
          for (size_t leftColRemainIdx = leftColBlockNumb * 8; leftColRemainIdx < colsInLeft; leftColRemainIdx++) {
            __m256 left0 = _mm256_set1_ps(left[(destRowIdx + 0) * colsInLeft + leftColRemainIdx]);
            __m256 left1 = _mm256_set1_ps(left[(destRowIdx + 1) * colsInLeft + leftColRemainIdx]);
            __m256 left2 = _mm256_set1_ps(left[(destRowIdx + 2) * colsInLeft + leftColRemainIdx]);
            __m256 left3 = _mm256_set1_ps(left[(destRowIdx + 3) * colsInLeft + leftColRemainIdx]);
            __m256 left4 = _mm256_set1_ps(left[(destRowIdx + 4) * colsInLeft + leftColRemainIdx]);
            __m256 left5 = _mm256_set1_ps(left[(destRowIdx + 5) * colsInLeft + leftColRemainIdx]);
            __m256 left6 = _mm256_set1_ps(left[(destRowIdx + 6) * colsInLeft + leftColRemainIdx]);
            __m256 left7 = _mm256_set1_ps(left[(destRowIdx + 7) * colsInLeft + leftColRemainIdx]);

            dest[0] = _mm256_fmadd_ps(left0, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[0]);
            dest[1] = _mm256_fmadd_ps(left1, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[1]);
            dest[2] = _mm256_fmadd_ps(left2, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[2]);
            dest[3] = _mm256_fmadd_ps(left3, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[3]);
            dest[4] = _mm256_fmadd_ps(left4, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[4]);
            dest[5] = _mm256_fmadd_ps(left5, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[5]);
            dest[6] = _mm256_fmadd_ps(left6, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[6]);
            dest[7] = _mm256_fmadd_ps(left7, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[7]);
          }
          __m256 resMultiplier = _mm256_set1_ps(alpha);
          _mm256_storeu_ps(&result[(destRowIdx + 0) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[0]));
          _mm256_storeu_ps(&result[(destRowIdx + 1) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[1]));
          _mm256_storeu_ps(&result[(destRowIdx + 2) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[2]));
          _mm256_storeu_ps(&result[(destRowIdx + 3) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[3]));
          _mm256_storeu_ps(&result[(destRowIdx + 4) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[4]));
          _mm256_storeu_ps(&result[(destRowIdx + 5) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[5]));
          _mm256_storeu_ps(&result[(destRowIdx + 6) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[6]));
          _mm256_storeu_ps(&result[(destRowIdx + 7) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[7]));
        }
      }

      // remained cols
      for (size_t destRowIdx = 0; destRowIdx < rowsInLeft; destRowIdx++) {
        for (size_t destColRemainIdx = destColBlockNumb * 8; destColRemainIdx < colsInRight; destColRemainIdx++) {
          float sum = 0;
          for (size_t leftColIdx = 0; leftColIdx < colsInLeft; leftColIdx++) {
            sum += left[destRowIdx * colsInLeft + leftColIdx] * right[leftColIdx * colsInRight + destColRemainIdx];
          }
          result[destRowIdx * colsInRight + destColRemainIdx] *= beta;
          result[destRowIdx * colsInRight + destColRemainIdx] = alpha * sum;
        }
      }

      // remained rows
      for (size_t destRowRemainIdx = destRowBlockNumb * 8; destRowRemainIdx < rowsInLeft; destRowRemainIdx++) {
        for (size_t destColIdx = 0; destColIdx < colsInRight - destColRemains; destColIdx++) {
          float sum = 0;
          for (size_t leftColIdx = 0; leftColIdx < colsInLeft; leftColIdx++) {
            sum += left[destRowRemainIdx * colsInLeft + leftColIdx] * right[leftColIdx * colsInRight + destColIdx];
          }
          result[destRowRemainIdx * colsInRight + destColIdx] *= beta;
          result[destRowRemainIdx * colsInRight + destColIdx] = alpha * sum;
        }
      }

    } else if (transLeft && !transRight) {
      
      size_t destRowBlockNumb = colsInLeft / 8;
      // size_t destRowRemains = colsInLeft % 7;
      size_t destColBlockNumb = colsInRight / 8;
      size_t destColRemains = colsInRight & 7;
      size_t leftColBlockNumb = rowsInLeft / 8;
      size_t leftColRemains = rowsInLeft & 7;

      for (size_t destRowIdx = 0; destRowIdx < destRowBlockNumb * 8; destRowIdx += 8) {
        for (size_t destColIdx = 0; destColIdx < destColBlockNumb * 8; destColIdx += 8) {
  
          __m256 multiplier = _mm256_set1_ps(beta);
          __m256 dest[8] = {
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 0) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 1) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 2) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 3) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 4) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 5) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 6) * colsInRight + destColIdx])),
            _mm256_mul_ps(multiplier, _mm256_loadu_ps(&result[(destRowIdx + 7) * colsInRight + destColIdx])),
          };

          for (size_t leftColIdx = 0; leftColIdx < leftColBlockNumb * 8; leftColIdx += 8) {
            __m256 right_[8] = {
              _mm256_loadu_ps(&right[(leftColIdx + 0) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 1) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 2) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 3) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 4) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 5) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 6) * colsInRight + destColIdx]),
              _mm256_loadu_ps(&right[(leftColIdx + 7) * colsInRight + destColIdx]),
            };

            for (size_t leftRowIdx = destRowIdx; leftRowIdx < destRowIdx + 8; leftRowIdx++) {
              
              __m256 left0 = _mm256_set1_ps(left[(leftColIdx + 0) * colsInLeft + leftRowIdx]);
              __m256 left1 = _mm256_set1_ps(left[(leftColIdx + 1) * colsInLeft + leftRowIdx]);
              __m256 left2 = _mm256_set1_ps(left[(leftColIdx + 2) * colsInLeft + leftRowIdx]);
              __m256 left3 = _mm256_set1_ps(left[(leftColIdx + 3) * colsInLeft + leftRowIdx]);
              __m256 left4 = _mm256_set1_ps(left[(leftColIdx + 4) * colsInLeft + leftRowIdx]);
              __m256 left5 = _mm256_set1_ps(left[(leftColIdx + 5) * colsInLeft + leftRowIdx]);
              __m256 left6 = _mm256_set1_ps(left[(leftColIdx + 6) * colsInLeft + leftRowIdx]);
              __m256 left7 = _mm256_set1_ps(left[(leftColIdx + 7) * colsInLeft + leftRowIdx]);

              size_t destIdx = leftRowIdx - destRowIdx;
              dest[destIdx] = _mm256_fmadd_ps(left0, right_[0], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left1, right_[1], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left2, right_[2], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left3, right_[3], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left4, right_[4], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left5, right_[5], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left6, right_[6], dest[destIdx]);
              dest[destIdx] = _mm256_fmadd_ps(left7, right_[7], dest[destIdx]);
            }
          }

          __m256 right_[leftColRemains];
          for (size_t leftColRemainIdx = leftColBlockNumb * 8; leftColRemainIdx < rowsInLeft; leftColRemainIdx++) {
            right_[leftColRemainIdx - leftColBlockNumb * 8] = _mm256_loadu_ps(&right[leftColRemainIdx * colsInRight + destColIdx]);
          }
          for (size_t leftColRemainIdx = leftColBlockNumb * 8; leftColRemainIdx < rowsInLeft; leftColRemainIdx++) {
            __m256 left0 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 0]);
            __m256 left1 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 1]);
            __m256 left2 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 2]);
            __m256 left3 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 3]);
            __m256 left4 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 4]);
            __m256 left5 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 5]);
            __m256 left6 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 6]);
            __m256 left7 = _mm256_set1_ps(left[leftColRemainIdx * colsInLeft + destRowIdx + 7]);

            dest[0] = _mm256_fmadd_ps(left0, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[0]);
            dest[1] = _mm256_fmadd_ps(left1, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[1]);
            dest[2] = _mm256_fmadd_ps(left2, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[2]);
            dest[3] = _mm256_fmadd_ps(left3, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[3]);
            dest[4] = _mm256_fmadd_ps(left4, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[4]);
            dest[5] = _mm256_fmadd_ps(left5, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[5]);
            dest[6] = _mm256_fmadd_ps(left6, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[6]);
            dest[7] = _mm256_fmadd_ps(left7, right_[leftColRemainIdx - leftColBlockNumb * 8], dest[7]);
          }

          __m256 resMultiplier = _mm256_set1_ps(alpha);
          _mm256_storeu_ps(&result[(destRowIdx + 0) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[0]));
          _mm256_storeu_ps(&result[(destRowIdx + 1) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[1]));
          _mm256_storeu_ps(&result[(destRowIdx + 2) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[2]));
          _mm256_storeu_ps(&result[(destRowIdx + 3) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[3]));
          _mm256_storeu_ps(&result[(destRowIdx + 4) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[4]));
          _mm256_storeu_ps(&result[(destRowIdx + 5) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[5]));
          _mm256_storeu_ps(&result[(destRowIdx + 6) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[6]));
          _mm256_storeu_ps(&result[(destRowIdx + 7) * colsInRight + destColIdx], _mm256_mul_ps(resMultiplier, dest[7]));
        }
      }

      // remained cols
      for (size_t destRowIdx = 0; destRowIdx < colsInLeft; destRowIdx++) {
        for (size_t destColRemainIdx = destColBlockNumb * 8; destColRemainIdx < colsInRight; destColRemainIdx++) {
          float sum = 0;
          for (size_t leftColIdx = 0; leftColIdx < rowsInLeft; leftColIdx++) {
            sum += left[leftColIdx * colsInLeft + destRowIdx] * right[leftColIdx * colsInRight + destColRemainIdx];
          }
          result[destRowIdx * colsInRight + destColRemainIdx] *= beta;
          result[destRowIdx * colsInRight + destColRemainIdx] += alpha * sum;
        }
      }

      // remained rows
      for (size_t destRowRemainIdx = destRowBlockNumb * 8; destRowRemainIdx < colsInLeft; destRowRemainIdx++) {
        for (size_t destColIdx = 0; destColIdx < colsInRight - destColRemains; destColIdx++) {
          float sum = 0;
          for (size_t leftColIdx = 0; leftColIdx < rowsInLeft; leftColIdx++) {
            sum += left[leftColIdx * colsInLeft + destRowRemainIdx] * right[leftColIdx * colsInRight + destColIdx];
          }
          result[destRowRemainIdx * colsInRight + destColIdx] *= beta;
          result[destRowRemainIdx * colsInRight + destColIdx] += alpha * sum;
        }
      }

    } else if (!transLeft && transRight) {
      for (size_t destRowIdx = 0; destRowIdx < rowsInLeft; destRowIdx++) {
        for (size_t destColIdx = 0; destColIdx < rowsInRight; destColIdx++) {
          const float* leftRow = &left[destRowIdx * colsInLeft];
          const float* rightRow = &right[destColIdx * colsInRight];
          result[destRowIdx * rowsInRight + destColIdx] *= beta;
          result[destRowIdx * rowsInRight + destColIdx] += alpha * dot(leftRow, rightRow, colsInLeft);          
        }
      }
    }
  }

  }
 }
}