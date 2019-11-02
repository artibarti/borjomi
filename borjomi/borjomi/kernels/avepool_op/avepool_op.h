#pragma once

#include "borjomi/types/types.h"

#include "borjomi/kernels/avepool_op/avepool_op_forward_internal.h"
#include "borjomi/kernels/avepool_op/avepool_op_backward_internal.h"

namespace borjomi {

  void averagePoolForwardOp(engine_t engine, const matrix_t& inData, matrix_t& outData,
    const shape3d_t& inShape, const shape3d_t& outShape, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::averagePoolInternalForward(inData, outData, inShape, outShape, poolingSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }

  void averagePoolBackwardOp(engine_t engine, matrix_t& prevDelta, const matrix_t& currDelta,
    const shape3d_t& inShape, const shape3d_t& outShape, size_t poolingSize) {

    if (engine == engine_t::internal) {
      kernels::averagePoolInternalBackward(prevDelta, currDelta, inShape, outShape, poolingSize);
    } else {
      throw BorjomiRuntimeException("Engine is not supported: " + toString(engine)); 
    }
  }
}