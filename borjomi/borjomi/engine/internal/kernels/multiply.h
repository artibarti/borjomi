#pragma once

#include <iostream>
#include <limits>

namespace borjomi {
 namespace engine {
  namespace internal {

  void multiply(float alpha, bool transLeft, size_t rowsInLeft, size_t colsInLeft, const float* left,
    bool transRight, size_t rowsInRight, size_t colsInRight, const float* right, float beta, float* result) {

    if (transLeft && transRight) {
      for(size_t colIdx = 0; colIdx < colsInRight; colIdx++) {
        for (size_t rowIdx = 0; rowIdx < rowsInLeft; rowIdx++) {
          float sum = 0;
          size_t resultIdx = rowIdx * colsInRight + colIdx;
          for (size_t c = 0; c < rowsInLeft; c++) {
            sum += alpha * left[c * colsInLeft + rowIdx] * right[colIdx * colsInRight + c];
          }
          result[resultIdx] *= beta;
          result[resultIdx] += sum;
        }
      }
    } else if (transLeft) {
      for(size_t colIdx = 0; colIdx < colsInRight; colIdx++) {
        for (size_t rowIdx = 0; rowIdx < colsInLeft; rowIdx++) {
          float sum = 0;
          size_t resultIdx = rowIdx * colsInRight + colIdx;
          for (size_t c = 0; c < rowsInLeft; c++) {
            sum += alpha * left[c * colsInLeft + rowIdx] * right[c * colsInRight + colIdx];
          }
          result[resultIdx] *= beta;
          result[resultIdx] += sum;
        }
      }
    } else if (transRight) {
      for(size_t colIdx = 0; colIdx < rowsInRight; colIdx++) {
        for (size_t rowIdx = 0; rowIdx < rowsInLeft; rowIdx++) {
          float sum = 0;
          size_t resultIdx = rowIdx * rowsInRight + colIdx;
          for (size_t c = 0; c < colsInLeft; c++) {
            sum += alpha * left[rowIdx * colsInLeft + c] * right[colIdx * colsInRight + c];
          }
          result[resultIdx] *= beta;
          result[resultIdx] += sum;
        }
      }
    } else {
      for(size_t colIdx = 0; colIdx < colsInRight; colIdx++) {
        for (size_t rowIdx = 0; rowIdx < rowsInLeft; rowIdx++) {
          float sum = 0;
          size_t resultIdx = rowIdx * colsInRight + colIdx;
          for (size_t c = 0; c < colsInLeft; c++) {
            sum += alpha * left[rowIdx * colsInLeft + c] * right[c * colsInRight + colIdx];
          }
          result[resultIdx] *= beta;
          result[resultIdx] += sum;
        }
      }
    }
  }

  }
 }  
}