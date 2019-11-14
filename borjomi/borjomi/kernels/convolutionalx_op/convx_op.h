#pragma once

#include "borjomi/kernels/convolutionalx_op/convx_op_forward_internal.h"
#include "borjomi/kernels/convolutionalx_op/convx_op_backward_internal.h"

namespace borjomi {

  void convxForwardOp(engine_t engine, const matrix_t& inData,
    const matrix_t& weights, const matrix_t& bias, matrix_t& outData,
    const shape3d_t& inShape, const shape3d_t& outShape, const shape3d_t& weightShape) {

    if (engine == engine_t::internal) {
      kernels::convxForwardInternal(inData, weights, bias,
        outData, inShape, outShape, weightShape);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }

  void convxBackwardOp(engine_t engine, const matrix_t& prevOut, const matrix_t& weights,
    matrix_t& dWeights, matrix_t& dBias, const matrix_t& currDelta, matrix_t& prevDelta,
    const shape3d_t& weightShape, const shape3d_t& inShape, const shape3d_t& outShape) {

    if (engine == engine_t::internal) {
      kernels::convxBackwardInternal(prevOut, weights, dWeights, dBias, currDelta,
        prevDelta, weightShape, inShape, outShape);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }
}