#pragma once

#include <iostream>
#include <limits>

namespace borjomi {
 namespace engine {
  namespace internal {

  void fill(float* dest, size_t destSize, float value) {
    for (size_t idx = 0; idx < destSize; idx++) {
      dest[idx] = value;
    }
  }

  }
 }  
}