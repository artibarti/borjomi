#pragma once

namespace borjomi {

struct shape3d_t {
  
 public:
  size_t rows_, cols_, channels_, area_, size_;

  shape3d_t(size_t rows, size_t cols, size_t channels)
    : rows_(rows), cols_(cols), channels_(channels), area_(rows * cols), size_(rows * cols * channels) {}
  
  shape3d_t() : rows_(0), cols_(0), channels_(0), area_(0), size_(0) {}

  size_t getIndex(size_t rowIdx, size_t colIdx, size_t channelIdx) const {
    return (rows_ * channelIdx + rowIdx) * cols_ + colIdx;
  }

  size_t area() const {
    return area_;
  }

  size_t size() const {
    return size_;
  }

	bool operator==(const shape3d_t& other) const {
		return (rows_ == other.rows_ && cols_ == other.cols_ && channels_ == other.channels_);
	}    

	bool operator!=(const shape3d_t& other) const {
		return !(rows_ == other.rows_ && cols_ == other.cols_ && channels_ == other.channels_);
	}
};

using shape3d_t = shape3d_t;

template <typename Stream>
Stream& operator<<(Stream& stream, const shape3d_t& shape) {
  stream << shape.rows_ << "x" << shape.cols_ << "x" << shape.channels_;
  return stream;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const shape3d_t& shape) {
  stream << shape.rows_ << "x" << shape.cols_ << "x" << shape.channels_;
  return stream;
}

}