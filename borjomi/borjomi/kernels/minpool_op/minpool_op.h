#pragma once

#include "borjomi/types/types.h"

#include "borjomi/kernels/minpool_op/minpool_op_forward_internal.h"
#include "borjomi/kernels/minpool_op/minpool_op_backward_internal.h"

namespace borjomi {

  void minPoolForwardOp(engine_t engine, const matrix_t& inData, matrix_t& outData,
    const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& minIndices, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::minpoolInternalForward(inData, outData, inShape, outShape, minIndices, poolingSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }

  void minPoolBackwardOp(engine_t engine, matrix_t& prevDelta, const matrix_t& currDelta,
    const shape3d_t& inShape, const shape3d_t& outShape, matrix_i& minIndices, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::minpoolInternalBackward(prevDelta, currDelta, inShape, outShape, minIndices, poolingSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }
}