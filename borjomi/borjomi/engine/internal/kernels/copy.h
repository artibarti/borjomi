#pragma once

#include <iostream>
#include <limits>

namespace borjomi {
 namespace engine {
  namespace internal {

  void copy(float alpha, bool transSource, size_t rowsInSource, size_t colsInSource,
    const float* source, size_t rowsInDest, size_t colsInDest, float* dest) {

    if (transSource) {
      for (size_t dstRowIdx = 0; dstRowIdx < rowsInDest; dstRowIdx += colsInSource) {
        for (size_t dstColIdx = 0; dstColIdx < colsInDest; dstColIdx += rowsInSource) {
          for (size_t srcRowIdx = 0; srcRowIdx < rowsInSource; srcRowIdx++) {
            for (size_t srcColIdx = 0; srcColIdx < colsInSource; srcColIdx++) {
              size_t dstIdx = (dstRowIdx + srcColIdx) * colsInDest + dstColIdx + srcRowIdx;
              dest[dstIdx] *= alpha;
              dest[dstIdx] += source[srcRowIdx * colsInSource + srcColIdx];
            }
          }
        }
      }
    } else {
      for (size_t dstRowIdx = 0; dstRowIdx < rowsInDest; dstRowIdx += rowsInSource) {
        for (size_t dstColIdx = 0; dstColIdx < colsInDest; dstColIdx += colsInSource) {
          for (size_t srcRowIdx = 0; srcRowIdx < rowsInSource; srcRowIdx++) {
            for (size_t srcColIdx = 0; srcColIdx < colsInSource; srcColIdx++) {
              size_t dstIdx = (dstRowIdx + srcRowIdx) * colsInDest + dstColIdx + srcColIdx;
              dest[dstIdx] *= alpha;
              dest[dstIdx] += source[srcRowIdx * colsInSource + srcColIdx];
            }
          }
        }
      }
    }
  }

  }
 }  
}