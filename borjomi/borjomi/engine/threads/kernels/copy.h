#pragma once

#include "borjomi/engine/threads/utils/parallel_loop.h"
#include "borjomi/engine/internal/kernels/kernels.h"

namespace borjomi {
 namespace engine {
  namespace threads {

  void copy(float alpha, bool transSource, size_t rowsInSource, size_t colsInSource,
    const float* source, size_t rowsInDest, size_t colsInDest, float* dest) {

    if (transSource) {
      threads::parallelized2DLoop(rowsInSource, colsInSource, 1, 1, [&](size_t srcRowIdx, size_t srcColIdx) {
          for (size_t dstRowIdx = 0; dstRowIdx < rowsInDest; dstRowIdx += colsInSource) {
            for (size_t dstColIdx = 0; dstColIdx < colsInDest; dstColIdx += rowsInSource) {
              size_t dstIdx = (dstRowIdx + srcColIdx) * colsInDest + dstColIdx + srcRowIdx;
              dest[dstIdx] *= alpha;
              dest[dstIdx] += source[srcRowIdx * colsInSource + srcColIdx];
            }
          }
      });
    } else {
      threads::parallelized2DLoop(rowsInSource, colsInSource, 1, 1, [&](size_t srcRowIdx, size_t srcColIdx) {
        for (size_t dstRowIdx = 0; dstRowIdx < rowsInDest; dstRowIdx += rowsInSource) {
          for (size_t dstColIdx = 0; dstColIdx < colsInDest; dstColIdx += colsInSource) {
            size_t dstIdx = (dstRowIdx + srcRowIdx) * colsInDest + dstColIdx + srcColIdx;
            dest[dstIdx] *= alpha;
            dest[dstIdx] += source[srcRowIdx * colsInSource + srcColIdx];
          }
        }
      });
    }
  }

  }
 }  
}