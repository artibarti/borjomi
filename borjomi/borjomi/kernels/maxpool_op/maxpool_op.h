#pragma once

#include "borjomi/types/types.h"

#include "borjomi/kernels/maxpool_op/maxpool_op_forward_internal.h"
#include "borjomi/kernels/maxpool_op/maxpool_op_backward_internal.h"

#ifdef USE_THREADS
  #include "borjomi/kernels/maxpool_op/maxpool_op_forward_threads.h"
  #include "borjomi/kernels/maxpool_op/maxpool_op_backward_threads.h"
#endif

namespace borjomi {

  void maxPoolForwardOp(engine_t engine, const matrix_t& inData, matrix_t& outData,
    const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& maxIndices, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::maxpoolForwardInternal(inData, outData, inShape, outShape, maxIndices, poolingSize);
    } else if (engine == engine_t::threads) {
      #ifdef USE_THREADS
        kernels::maxpoolForwardThreads(inData, outData, inShape, outShape, maxIndices, poolingSize);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with thread support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }

  void maxPoolBackwardOp(engine_t engine, matrix_t& prevDelta, const matrix_t& currDelta,
    const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& maxIndices, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::maxpoolBackwardInternal(prevDelta, currDelta, inShape, outShape, maxIndices, poolingSize);
    } else if (engine == engine_t::threads) {
      #ifdef USE_THREADS
        kernels::maxpoolBackwardThreads(prevDelta, currDelta, inShape, outShape, maxIndices, poolingSize);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with thread support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }
}