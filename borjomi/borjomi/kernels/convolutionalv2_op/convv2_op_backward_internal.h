#pragma once

#include "borjomi/kernels/params/conv_params.h"

namespace borjomi {
namespace kernels {

void convv2BackwardInternal(const matrix_t& prevOut, const matrix_t& W,
  matrix_t& dW, matrix_t& db, const matrix_t& currDelta, matrix_t& prevDelta) {

}

}
}