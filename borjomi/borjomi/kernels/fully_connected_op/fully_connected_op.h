#pragma once

#include "borjomi/types/types.h"

#include "borjomi/kernels/fully_connected_op/fully_connected_op_forward_internal.h"
#include "borjomi/kernels/fully_connected_op/fully_connected_op_backward_internal.h"

#include "borjomi/kernels/fully_connected_op/fully_connected_op_forward_threads.h"
#include "borjomi/kernels/fully_connected_op/fully_connected_op_backward_threads.h"

#ifdef CNN_USE_AVX2
  #include "borjomi/kernels/fully_connected_op/fully_connected_op_forward_avx.h"
  #include "borjomi/kernels/fully_connected_op/fully_connected_op_backward_avx.h"
#endif

namespace borjomi {

  void fullyConnectedForwardOp(engine_t engine, const matrix_t& inData,
    const matrix_t& weights, const matrix_t& bias, matrix_t& outData) {

    if (engine == engine_t::internal) {
      kernels::fullyConnectedForwardInternal(inData, weights, bias, outData);
    } else if (engine == engine_t::threads) {
        kernels::fullyConnectedForwardThreads(inData, weights, bias, outData);
    } else if (engine == engine_t::avx) {
      #ifdef CNN_USE_AVX2
        kernels::fullyConnectedForwardAvx(inData, weights, bias, outData);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with AVX support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }

  void fullyConnectedBackwardOp(engine_t engine, const matrix_t& prevOut, const matrix_t& weights,
    matrix_t& dWeights, matrix_t& dBias, const matrix_t& currDelta, matrix_t& prevDelta) {

    if (engine == engine_t::internal) {
      kernels::fullyConnectedBackwardInternal(prevOut, weights, dWeights, dBias, currDelta, prevDelta);
    } else if (engine == engine_t::threads) {
        kernels::fullyConnectedBackwardThreads(prevOut, weights, dWeights, dBias, currDelta, prevDelta);
    } else if (engine == engine_t::avx) {
      #ifdef CNN_USE_AVX2
        kernels::fullyConnectedBackwardAvx(prevOut, weights, dWeights, dBias, currDelta, prevDelta);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with AVX support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }
}