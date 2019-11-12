#pragma once

#include <memory>
#include <vector>

#include "borjomi/types/allocator_t.h"
#include "borjomi/types/shape2d_t.h"

namespace borjomi {

template<typename T>
class Matrix {

  private:
    std::vector<T, AlignedAllocator<T, 64>> matrixData_;
    shape2d_t shape_;

  public:
    Matrix();
    Matrix(Matrix<T>&& other);
    Matrix(const Matrix<T>& matrix);
    Matrix(shape2d_t shape, T value = 0);
    Matrix(size_t rows, size_t cols, T value = 0);
    Matrix<T>& operator= (const Matrix<T>& matrix);

    shape2d_t shape() const;
    size_t size() const;
    size_t rows() const;
    size_t cols() const;

    void reshape(size_t rows, size_t cols, T value = 0);
    void reshape(const shape2d_t shape, T value = 0);

    T* data();
    const T* data() const;

    T& at(size_t idx);
    const T& at(size_t idx) const;
    T& at(size_t rowIdx, size_t colIdx);
    const T& at(size_t rowIdx, size_t colIdx) const;

    void setRow(size_t rowIdx, const std::vector<T>& row);
    void setColumn(size_t colIdx, const std::vector<T>& col);

    bool isEmpty() const;

    bool isMultiplicableWith(const Matrix<T>& other) const;
    bool hasSameDimensionWith(const Matrix<T>& other) const;
};

template<typename T>
Matrix<T>::Matrix() {
  reshape(0,0);
}

template<typename T>
Matrix<T>::Matrix(shape2d_t shape, T value) {
  reshape(shape.rows_, shape.cols_, value);
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T value) {
  reshape(rows, cols, value);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) {
  reshape(other.shape_);
  for (size_t elementIdx = 0; elementIdx < other.size(); elementIdx++) {
    matrixData_[elementIdx] = other.matrixData_[elementIdx];
  }
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) {
  shape_ = other.shape_;
  matrixData_ = other.matrixData_;
  other.reshape(0,0);
}

template<typename T>
Matrix<T>& Matrix<T>::operator= (const Matrix<T>& other) {
  reshape(other.shape_.rows_, other.shape_.cols_);
  for (size_t elementIdx = 0; elementIdx < other.size(); elementIdx++) {
    matrixData_[elementIdx] = other.matrixData_[elementIdx];
  }
  return *this;
}

template<typename T>
shape2d_t Matrix<T>::shape() const {
  return shape_;
}

template<typename T>
size_t Matrix<T>::size() const {
  return shape_.size();
}

template<typename T>
size_t Matrix<T>::rows() const {
  return shape_.rows_;
}

template<typename T>
size_t Matrix<T>::cols() const {
  return shape_.cols_;
}

template<typename T>
void Matrix<T>::reshape(size_t rows, size_t cols, T value) {  
  reshape(shape2d_t(rows, cols), value);
}

template<typename T>
void Matrix<T>::reshape(const shape2d_t shape, T value) {
  if (shape_.size() != shape.size()) {
    shape_ = shape;
    matrixData_.resize(shape_.size(), value);
  } else {
    shape_ = shape;
  }
}

template<typename T>
T* Matrix<T>::data() {
  return matrixData_.data();
}

template<typename T>
const T* Matrix<T>::data() const {
  return matrixData_.data();
}

template<typename T>
T& Matrix<T>::at(size_t idx) {
  return matrixData_[idx];
}

template<typename T>
const T& Matrix<T>::at(size_t idx) const {
  return matrixData_[idx];
}

template<typename T>
T& Matrix<T>::at(size_t rowIdx, size_t colIdx) {
  return matrixData_[rowIdx * shape_.cols_ + colIdx];
}

template<typename T>
const T& Matrix<T>::at(size_t rowIdx, size_t colIdx) const {
  return matrixData_[rowIdx * shape_.cols_ + colIdx];
}

template<typename T>
void Matrix<T>::setRow(size_t rowIdx, const std::vector<T>& row) {
  if (rowIdx < shape_.rows_ && row.size() == shape_.cols_) {
    for (size_t colIdx = 0; colIdx < shape_.cols_; colIdx++) {
      matrixData_[rowIdx * shape_.cols_ + colIdx] = row[colIdx];
    }
  } else {
    throw std::out_of_range("Index out of bounds");
  }
}

template<typename T>
void Matrix<T>::setColumn(size_t colIdx, const std::vector<T>& col) {
  if (colIdx < shape_.cols_ && col.size() == shape_.rows_) {
    for (size_t rowIdx = 0; rowIdx < shape_.rows_; rowIdx++) {
      matrixData_[rowIdx * shape_.cols_ + colIdx] = col[rowIdx];
    }
  } else {
    throw std::out_of_range("Index out of bounds");
  }
}

template<typename T>
bool Matrix<T>::isEmpty() const {
  return shape_ == shape2d_t(0,0);
}

template<typename T>
bool Matrix<T>::isMultiplicableWith(const Matrix<T>& other) const {
  return shape_.cols_ == other.shape_.rows_;
}
    
template<typename T>
bool Matrix<T>::hasSameDimensionWith(const Matrix<T>& other) const {
  return shape_.rows_ == other.shape_.rows_ && shape_.cols_ == other.shape_.cols_;
}

using matrix_t = Matrix<float>;
using matrix_i = Matrix<size_t>;

}