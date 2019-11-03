#pragma once

#include "borjomi/engine/internal/kernels/kernels.h"

#ifdef USE_THREADS
  #include "borjomi/engine/threads/kernels/kernels.h"
#endif

#ifdef USE_CUDA
  #include "borjomi/engine/cuda/kernels/kernels.h"
#endif

#ifdef USE_AVX2
  #include "borjomi/engine/avx2/kernels/kernels.h"
#endif