#pragma once

#include "borjomi/kernels/convolutionalv2_op/convv2_op_forward_internal.h"
#include "borjomi/kernels/convolutionalv2_op/convv2_op_backward_internal.h"

namespace borjomi {

  void convv2ForwardOp(engine_t engine, const matrix_t& inData,
    const matrix_t& weights, const matrix_t& bias, matrix_t& outData,
    const shape3d_t& inShape, const shape3d_t& outShape, size_t kernelSize) {

    if (engine == engine_t::internal) {
      kernels::convv2ForwardInternal(inData, weights, bias,
        outData, inShape, outShape, kernelSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }

  void convv2BackwardOp(engine_t engine, const matrix_t& prevOut, const matrix_t& reorganizedPrevOut,
    const matrix_t& W, matrix_t& dW, matrix_t& db, const matrix_t& currDelta,
    matrix_t& reorganizedPrevDelta, const shape3d_t& weightShape, const shape3d_t& inShape,
    const shape3d_t& reorderedInShape, const shape3d_t& outShape, size_t kernelSize) {

    if (engine == engine_t::internal) {
      kernels::convv2BackwardInternal(prevOut, reorganizedPrevOut, W, dW, db,
        currDelta, reorganizedPrevDelta, weightShape, inShape, reorderedInShape, outShape, kernelSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine));
    }
  }
}