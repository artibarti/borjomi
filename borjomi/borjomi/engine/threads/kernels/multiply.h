#pragma once

#include "borjomi/engine/threads/utils/parallel_loop.h"
#include "borjomi/engine/internal/kernels/kernels.h"

namespace borjomi {
 namespace engine {
  namespace threads {

  void multiply(float alpha, bool transLeft, size_t rowsInLeft, size_t colsInLeft, const float* left,
    bool transRight, size_t rowsInRight, size_t colsInRight, const float* right, float beta, float* result) {

    if (transLeft && transRight) {
      threads::parallelized2DLoop(colsInLeft, rowsInRight, 1, 1, [&](size_t rowIdx, size_t colIdx) {
        float sum = 0;
        size_t resultIdx = rowIdx * colsInRight + colIdx;
        for (size_t c = 0; c < rowsInLeft; c++) {
          sum += alpha * left[c * colsInLeft + rowIdx] * right[colIdx * colsInRight + c];
        }
        result[resultIdx] *= beta;
        result[resultIdx] += sum;
      });
    } else if (transLeft) {
      threads::parallelized2DLoop(colsInLeft, colsInRight, 1, 1, [&](size_t rowIdx, size_t colIdx) {
        float sum = 0;
        size_t resultIdx = rowIdx * colsInRight + colIdx;
        for (size_t c = 0; c < rowsInLeft; c++) {
          sum += alpha * left[c * colsInLeft + rowIdx] * right[c * colsInRight + colIdx];
        }
        result[resultIdx] *= beta;
        result[resultIdx] += sum;
      });
    } else if (transRight) {
      threads::parallelized2DLoop(rowsInLeft, rowsInRight, 1, 1, [&](size_t rowIdx, size_t colIdx) {
        float sum = 0;
        size_t resultIdx = rowIdx * rowsInRight + colIdx;
        for (size_t c = 0; c < colsInLeft; c++) {
          sum += alpha * left[rowIdx * colsInLeft + c] * right[colIdx * colsInRight + c];
        }
        result[resultIdx] *= beta;
        result[resultIdx] += sum;
      });
    } else {
      threads::parallelized2DLoop(rowsInLeft, colsInRight, 1, 1, [&](size_t rowIdx, size_t colIdx) {
        float sum = 0;
        size_t resultIdx = rowIdx * colsInRight + colIdx;
        for (size_t c = 0; c < colsInLeft; c++) {
          sum += alpha * left[rowIdx * colsInLeft + c] * right[c * colsInRight + colIdx];
        }
        result[resultIdx] *= beta;
        result[resultIdx] += sum;
      });
    }
  }

  }
 }  
}