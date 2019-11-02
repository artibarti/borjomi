#pragma once

#include <iostream>

extern "C" {
  #include <mkl_cblas.h>
}

#include "borjomi/engine/internal/kernels/kernels.h"

namespace borjomi {
 namespace engine {
  namespace mkl {

  void multiply(float alpha, bool transLeft, size_t rowsInLeft, size_t colsInLeft, const float* left,
    bool transRight, size_t rowsInRight, size_t colsInRight, const float* right, float beta, float* result) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, out_size, in_size,
      alpha, input, in_size, weight, out_size, beta, output, out_size);    
  }
  
  }
 }
}