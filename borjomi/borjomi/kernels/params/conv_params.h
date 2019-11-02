/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <deque>
#include <vector>

#include "borjomi/kernels/params/params.h"

namespace borjomi {
namespace kernels {

class conv_params : public Params {
 public:
  shape3d_t in;
  shape3d_t in_padded;
  shape3d_t out;
  shape3d_t weight;
  padding pad_type;

  friend std::ostream &operator<<(std::ostream &o, const kernels::conv_params &param) {
    o << "in:        " << param.in << std::endl;
    o << "out:       " << param.out << std::endl;
    o << "in_padded: " << param.in_padded << std::endl;
    o << "weight:    " << param.weight << std::endl;
    return o;
  }
};

inline conv_params &Params::conv() {
  return *(static_cast<conv_params *>(this));
}

class Conv2dPadding {
 public:
  
  Conv2dPadding() {}
  explicit Conv2dPadding(const conv_params &params) : params_(params) {}

  void addPadding(const matrix_t& in, matrix_t& out, size_t paddingSize) {

    if (params_.pad_type == padding::valid) {
      return;
    }

    out.reshape(in.rows(), params_.in_padded.size(), float{0});
    for(size_t sample = 0; sample < out.rows(); sample++) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < params_.in.channels_; inputChannelIdx++) {
        for (size_t inputRowIdx = 0; inputRowIdx < params_.in.rows_; inputRowIdx++) {
          for (size_t inputColIdx = 0; inputColIdx < params_.in.cols_; inputColIdx++) {
            size_t outIdx = params_.in_padded.getIndex(paddingSize + inputRowIdx, paddingSize + inputColIdx, inputChannelIdx);
            size_t inIdx = params_.in.getIndex(inputRowIdx, inputColIdx, inputChannelIdx);
            out.at(sample, outIdx) = in.at(sample, inIdx);
          }
        }
      }
    }
  }

  void removePadding(const matrix_t& delta, matrix_t& delta_unpadded, size_t paddingSize) {

    if (params_.pad_type == padding::valid) {
      return;
    }

    delta_unpadded.reshape(delta.rows(), params_.in.size());
    for(size_t sample = 0; sample < delta_unpadded.rows(); sample++) {
      for (size_t inputChannelIdx = 0; inputChannelIdx < params_.in.channels_; inputChannelIdx++) {
        for (size_t inputRowIdx = 0; inputRowIdx < params_.in.rows_; inputRowIdx++) {
          for (size_t inputColIdx = 0; inputColIdx < params_.in.cols_; inputColIdx++) {
            size_t deltaIdx = params_.in_padded.getIndex(paddingSize + inputRowIdx, paddingSize + inputColIdx, inputChannelIdx);
            size_t deltaUnpaddedIdx = params_.in.getIndex(inputRowIdx, inputColIdx, inputChannelIdx);
            delta_unpadded.at(sample, deltaUnpaddedIdx) = delta.at(sample, deltaIdx);
          }
        }
      }
    }
  }

 private:
  conv_params params_;
};

}
}
