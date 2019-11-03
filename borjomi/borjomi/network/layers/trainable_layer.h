#pragma once

#include "borjomi/types/types.h"

namespace borjomi {

  class TrainableLayer : public Layer {

    private:
      bool hasBias_;

    public:
      TrainableLayer(size_t inSize, size_t outSize,
        bool hasBias = true, engine_t engine = engine_t::internal)
        : TrainableLayer(shape3d_t(1, inSize, 1), shape3d_t(1, outSize, 1), hasBias, engine) {}

      TrainableLayer(const shape3d_t& inShape, const shape3d_t& outShape,
        bool hasBias = false, engine_t engine = engine_t::internal)
        : Layer(inShape, outShape, false, true, engine) {

        registerEdge("weight", content_t::weight);
        hasBias_ = hasBias;
        if (hasBias_) {
          registerEdge("bias", content_t::bias);
        }
      }

      TrainableLayer(TrainableLayer&& other) : Layer(std::move(other)) {
        hasBias_ = other.hasBias_;
      }

      void setBatchSize(size_t batchSize = 1) override {
        reshapeEdge("incomingEdge", shape2d_t(batchSize, getInDataSize()));
        reshapeEdge("outgoingEdge", shape2d_t(batchSize, getOutDataSize()));
      }

      bool hasBias() {
        return hasBias_;
      }
  };
}