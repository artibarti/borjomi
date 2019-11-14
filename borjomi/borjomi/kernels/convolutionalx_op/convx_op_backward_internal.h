#pragma once

#include <algorithm>

namespace borjomi {
namespace kernels {

void convxCalculatePrevDeltaForOneSample(const float* currDelta,
  const matrix_t& weights, float* prevDelta, const shape3d_t& inShape,
  const shape3d_t& outShape, const shape3d_t& weightShape) {

  size_t kernelSize = weightShape.rows_;
  size_t padding = std::floor(kernelSize / 2.0f);

  for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
    for (size_t outRowIdx = 0; outRowIdx < outShape.rows_; outRowIdx++) {
      for (size_t outColIdx = 0; outColIdx < outShape.cols_; outColIdx++) {

        float currDeltaValue = currDelta[outShape.getIndex(outRowIdx, outColIdx, outChannelIdx)];

        size_t inRowBegin = std::max(int(outRowIdx - padding), 0);
        size_t inRowEnd = std::min(int(outRowIdx + padding), int(inShape.rows_ - 1));
        size_t nRows = inRowEnd - inRowBegin + 1;

        size_t inColBegin = std::max(int(outColIdx - padding), 0);
        size_t inColEnd = std::min(int(outColIdx + padding), int(inShape.cols_ - 1));
        size_t nCols = inColEnd - inColBegin + 1;

        size_t weightRowBegin = int(outRowIdx - padding) >= 0 ? 0 : kernelSize - nRows;
        size_t weightChannelBegin = int(outColIdx - padding) >= 0 ? 0 : kernelSize - nCols;

        for (size_t inChannelIdx = 0; inChannelIdx < inShape.channels_; inChannelIdx++) {

          for (size_t inColIdx = inColBegin, weightChannelIdx = weightChannelBegin;
            inColIdx <= inColEnd; inColIdx++, weightChannelIdx++) {

            for (size_t inRowIdx = inRowBegin, weightRowIdx = weightRowBegin;
              inRowIdx <= inRowEnd; inRowIdx++, weightRowIdx++) {

              prevDelta[inShape.getIndex(inRowIdx, inColIdx, inChannelIdx)]
                += currDeltaValue * weights.at(outChannelIdx,
                  weightShape.getIndex(weightRowIdx, inChannelIdx, weightChannelIdx));
            }
          }
        }
      }
    }
  }
}

void calculateWeightDeltaForOneSample(matrix_t& dWeights, const float* prevOut, const float* currDelta,
  const shape3d_t& inShape, const shape3d_t& outShape, const shape3d_t& weightShape) {

  size_t kernelSize = weightShape.rows_;
  size_t padding = std::floor(kernelSize / 2.0f);

  for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
    for (size_t weightChannelIdx = 0; weightChannelIdx < weightShape.channels_; weightChannelIdx++) {
      for (size_t weightRowIdx = 0; weightRowIdx < weightShape.rows_; weightRowIdx++) {
        for (size_t weightColIdx = 0; weightColIdx < weightShape.cols_; weightColIdx++) {

          float& dw = dWeights.at(outChannelIdx, weightShape.getIndex(weightRowIdx, weightColIdx, weightChannelIdx));

          size_t prevOutRowBeginIdx = std::max(int(weightRowIdx - padding), 0);
          size_t prevOutRowEndIdx = std::min(int(outShape.rows_ - 1),
            int(outShape.rows_ - 1 + weightRowIdx - padding));

          size_t prevOutColBeginIdx = std::max(int(weightChannelIdx - padding), 0);
          size_t prevOutColEndIdx = std::min(int(outShape.cols_ - 1),
            int(outShape.cols_ - 1 + weightChannelIdx - padding));

          size_t currDeltaRowBeginIdx = (outShape.rows_ - 1) - prevOutRowEndIdx;
          size_t currDeltaColBeginIdx = (outShape.cols_ - 1) - prevOutColEndIdx;

          float dot = 0;
          for (size_t prevOutRowIdx = prevOutRowBeginIdx, currDeltaRowIdx = currDeltaRowBeginIdx;
            prevOutRowIdx <= prevOutRowEndIdx; prevOutRowIdx++, currDeltaRowIdx++) {
            for (size_t prevOutColIdx = prevOutColBeginIdx, currDeltaColIdx = currDeltaColBeginIdx;
              prevOutColIdx <= prevOutColEndIdx; prevOutColIdx++, currDeltaColIdx++) {
              dot += currDelta[outShape.getIndex(currDeltaRowIdx, currDeltaColIdx, outChannelIdx)]
                * prevOut[inShape.getIndex(prevOutRowIdx, prevOutColIdx, weightColIdx)];
            }
          }
          dw += dot;
        }
      }
    }
  }  
}

void calculateBiasDeltaForOneSample(matrix_t& dBias,
  const float_t* currDelta, const shape3d_t& outShape) {

  for (size_t outChannelIdx = 0; outChannelIdx < outShape.channels_; outChannelIdx++) {
    const float *currDeltaChannelPtr = &currDelta[outShape.getIndex(0, 0, outChannelIdx)];
    const float *currDeltaNextChannelPtr = currDeltaChannelPtr + outShape.area();
    dBias.at(0, outChannelIdx) += std::accumulate(currDeltaChannelPtr, currDeltaNextChannelPtr, float{0});
  }
  
}

void convxBackwardInternal(const matrix_t& prevOut, const matrix_t& weights,
  matrix_t& dWeights, matrix_t& dBias, const matrix_t& currDelta, matrix_t& prevDelta,
  const shape3d_t& weightShape, const shape3d_t& inShape, const shape3d_t& outShape) {

  for (size_t sampleIdx = 0; sampleIdx < currDelta.rows(); sampleIdx++) {

    // prevDelta
    convxCalculatePrevDeltaForOneSample(&currDelta.at(sampleIdx, 0),
      weights, &prevDelta.at(sampleIdx, 0), inShape, outShape, weightShape);

    // dw
    calculateWeightDeltaForOneSample(dWeights, &prevOut.at(sampleIdx, 0),
      &currDelta.at(sampleIdx, 0), inShape, outShape, weightShape);

    // db
    if (!dBias.isEmpty()) {
      calculateBiasDeltaForOneSample(dBias,
        &currDelta.at(sampleIdx, 0), outShape);
    }
  }
}

}
}