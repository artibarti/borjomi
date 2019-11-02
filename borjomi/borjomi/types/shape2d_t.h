#pragma once

#include <iostream>

namespace borjomi {

struct Shape2d {

 public:
	size_t rows_, cols_, size_;

	Shape2d() : rows_(0), cols_(0), size_(0) {}
	Shape2d(size_t rows, size_t cols) : rows_(rows), cols_(cols), size_(rows * cols) {}

	void operator=(const Shape2d& other) {
		rows_ = other.rows_;
		cols_ = other.cols_;
	}

	bool operator==(const Shape2d& other) const {
		return (rows_ == other.rows_ && cols_ == other.cols_);
	}    

	bool operator!=(const Shape2d& other) const {
		return !(rows_ == other.rows_ && cols_ == other.cols_);
	}

	size_t size() const {
		return rows_ * cols_;
	}

	size_t rows() const {
		return rows_;
	}

	size_t cols() const {
		return cols_;
	}
};

template<typename Stream>
Stream &operator<<(Stream& stream, const Shape2d& shape) {
  stream << shape.rows() << "x" << shape.cols();
  return stream;
}

template <typename T>
std::ostream &operator<<(std::ostream& stream, const Shape2d& shape) {
  stream << shape.rows() << "x" << shape.cols();
  return stream;
}

using shape2d_t = Shape2d;

}