#pragma once

#include "borjomi/types/types.h"
#include "borjomi/engine/internal/kernels/kernels.h"

#include <immintrin.h>
#include "borjomi/engine/avx2/utils/utils2.h"

namespace borjomi {
 namespace engine {
  namespace avx {

  void fill(float* dest, size_t destSize, float value) {    
    size_t numberOfBlocks = destSize / 8;
    size_t remains = destSize % 8;
    __m256 val = _mm256_set1_ps(value);
    for (size_t blockIdx = 0; blockIdx < numberOfBlocks; blockIdx++) {
      _mm256_store_ps(&dest[blockIdx * 8], val);
    }
    for (size_t remainIdx = 0; remainIdx < remains; remainIdx++) {
      dest[numberOfBlocks * 8 + remainIdx] = value;
    }
  }

  }
 }
}