#pragma once

#include "borjomi/types/types.h"
#include "borjomi/network/layers/layer.h"

namespace borjomi {

  class PoolingLayer : public Layer {

    private:
      size_t poolingSize_;
      matrix_i indices_;

      static size_t calcOutDim(size_t inDim, size_t poolingSize) {
        float tmp = static_cast<float>(inDim) / poolingSize;
        return static_cast<int>(floor(tmp));
      }

    public:
      PoolingLayer(size_t inWidth, size_t inHeight, size_t inChannels, size_t poolingSize, engine_t engine = engine_t::internal)
       : Layer(shape3d_t(inHeight, inWidth, inChannels), shape3d_t(calcOutDim(inHeight, poolingSize),
         calcOutDim(inWidth, poolingSize), inChannels), engine), poolingSize_(poolingSize) {}

      PoolingLayer(PoolingLayer&& other) : Layer(std::move(other)) {
        indices_ = other.indices_;
        poolingSize_ = other.poolingSize_;
      }

      size_t poolingSize() {
        return poolingSize_;
      }

      matrix_i& getIndices() {
        return indices_;
      }

      void setBatchSize(size_t batchSize = 1) override {
        shape2d_t inShape(batchSize, getInDataSize());
        shape2d_t outShape(batchSize, getOutDataSize());
        reshapeEdge("incomingEdge", inShape);
        reshapeEdge("outgoingEdge", outShape);
        indices_.reshape(batchSize, getOutDataSize());
      }
  };
}