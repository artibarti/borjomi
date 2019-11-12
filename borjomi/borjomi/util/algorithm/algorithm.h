#pragma once

#include <vector>

#include "borjomi/types/types.h"

namespace borjomi {

void normalizeMatrix(const matrix_t& inputs, std::vector<matrix_t> &normalized) {
  for (size_t sampleIndex = 0; sampleIndex < inputs.rows(); sampleIndex++) {
    matrix_t m(1, inputs.cols());
    for (size_t colIdx = 0; colIdx < inputs.cols(); colIdx++) {
     m.at(0, colIdx) = inputs.at(sampleIndex, colIdx);
    }
    normalized.push_back(m);
  }
}

void convertLabelsToMatrix(const std::vector<label_t>& labels, matrix_t& matrix,
  size_t outDataSize, std::pair<float, float> targetValueRange) {
  
  matrix.reshape(labels.size(), outDataSize);
  for (size_t labelIndex = 0; labelIndex < labels.size(); labelIndex++) {
    if (labels[labelIndex] > outDataSize) {
      throw BorjomiRuntimeException("convertLabelsToMatrix: label is greater than output data size");
    }
    for (size_t colIndex = 0; colIndex < outDataSize; colIndex++) {
      matrix.at(labelIndex, colIndex) = targetValueRange.first;
    }
    matrix.at(labelIndex, labels[labelIndex]) = targetValueRange.second;
  }
}

size_t getIndexOfMaxElementInRow(const matrix_t& matrix, size_t rowIndex) {

  if (rowIndex >= matrix.rows()) {
    throw BorjomiRuntimeException("getMaxElementInRow: Index out of bounds");
  } else if (matrix.cols() == 0) {
    throw BorjomiRuntimeException("getMaxElementInRow: Matrix has null dimension");
  }

  float maxElement = matrix.at(rowIndex, 0);
  size_t maxIndex = 0;
  for (size_t colIndex = 1; colIndex < matrix.cols(); colIndex++) {
    if (matrix.at(rowIndex, colIndex) > maxElement) {
      maxElement = matrix.at(rowIndex, colIndex);
      maxIndex = colIndex;
    }
  }
  return maxIndex;
}

std::pair<int, float> compare(const matrix_t& one, const matrix_t& other) {
  
  if (one.shape() != other.shape()) {
    throw BorjomiRuntimeException("compare: Matrices are incompatible, shapes are different");
  }

  shape2d_t matrixShape = one.shape();
  std::pair<int, float> result(0, 0.0);
  for (size_t rowIdx = 0; rowIdx < matrixShape.rows_; rowIdx++) {
    for (size_t colIdx = 0; colIdx < matrixShape.cols_; colIdx++) {
      float distance = std::abs(one.at(rowIdx, colIdx) - other.at(rowIdx, colIdx));
      if (distance >= 0.01) {
        std::cout << one.at(rowIdx, colIdx) << "\t" << other.at(rowIdx, colIdx)
          << "\t" << rowIdx << "x" << colIdx << std::endl;
        result.first++;
        result.second += distance;
      }
    }
  }
  result.second /= std::max(result.first, 1);
  return result;
}

}