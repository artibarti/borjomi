#pragma once

#include "borjomi/kernels/convolutional_op/conv_op_forward_internal.h"
#include "borjomi/kernels/convolutional_op/conv_op_backward_internal.h"

#include "borjomi/kernels/convolutional_op/conv_op_forward_threads.h"
#include "borjomi/kernels/convolutional_op/conv_op_backward_threads.h"

#ifdef CNN_USE_AVX2
  #include "borjomi/kernels/convolutional_op/conv_op_forward_avx.h"
  #include "borjomi/kernels/convolutional_op/conv_op_backward_avx.h"
#endif

namespace borjomi {

  void convForwardOp(engine_t engine, const matrix_t& inData, const matrix_t& weights,
    const matrix_t& bias, matrix_t& outData, const kernels::conv_params& params) {

    if (engine == engine_t::internal) {
      kernels::convForwardInternal(inData, weights, bias, outData, params.in, params.in_padded, params.out, params.weight);
    } else if (engine == engine_t::threads) {
        kernels::convForwardThreads(inData, weights, bias, outData, params.in, params.in_padded, params.out, params.weight);
    } else if (engine == engine_t::avx) {
      #ifdef CNN_USE_AVX2
        kernels::convForwardAvx(inData, weights, bias, outData, params.in, params.in_padded, params.out, params.weight);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with AVX support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }

  void convBackwardOp(engine_t engine, const matrix_t& prevOut, const matrix_t& weights, matrix_t& dWeights,
    matrix_t& dBias, const matrix_t& currDelta, matrix_t& prevDelta, const kernels::conv_params &params) {

    if (engine == engine_t::internal) {
      kernels::convBackwardInternal(prevOut, weights, dWeights, dBias, currDelta, prevDelta, params.in, params.in_padded, params.out, params.weight);
    } else if (engine == engine_t::threads) {
        kernels::convBackwardThreads(prevOut, weights, dWeights, dBias, currDelta, prevDelta, params.in, params.in_padded, params.out, params.weight);
    } else if (engine == engine_t::avx) {
      #ifdef CNN_USE_AVX2
        kernels::convBackwardAvx(prevOut, weights, dWeights, dBias, currDelta, prevDelta, params.in, params.in_padded, params.out, params.weight);
      #else
        throw BorjomiRuntimeException("Borjomi was not built with AVX support");
      #endif
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }
}