#pragma once

#include "borjomi/engine/threads/utils/parallel_loop.h"
#include "borjomi/engine/internal/kernels/kernels.h"

namespace borjomi {
 namespace engine {
  namespace threads {

  void fill(float* dest, size_t destSize, float value) {
    if (destSize < 200) {
      internal::fill(dest, destSize, value);
      return;
    }
    threads::parallelized2DLoop(1, destSize, 1, 1, [&](size_t rowIdx, size_t colIdx) {
      dest[colIdx] = value;
    });
  }

  }
 }  
}