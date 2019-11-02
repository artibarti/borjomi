#pragma once

#include "borjomi/engine/internal/kernels/kernels.h"
#include "borjomi/engine/threads/kernels/kernels.h"

#ifdef USE_OPENCL
  #include "borjomi/engine/cuda/kernels/kernels.h"
#endif

#ifdef CNN_USE_AVX2
  #include "borjomi/engine/avx/kernels/kernels.h"
#endif