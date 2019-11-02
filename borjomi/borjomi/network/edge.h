#pragma once

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"

namespace borjomi {

class Edge {

 private:
  shape2d_t shape_;
  matrix_t data_;
  matrix_t gradient_;
  content_t contentType_;

 public:
  Edge(content_t contentType, const shape2d_t &shape = shape2d_t(0,0));

  matrix_t& getData();
  const matrix_t& getData() const;

  matrix_t& getGradient();
  const matrix_t& getGradient() const;

  content_t getContentType() const;

  const shape2d_t &shape() const;
  void reshape(const shape2d_t& shape);
};

typedef std::shared_ptr<Edge> edgeptr_t;

Edge::Edge(content_t contentType, const shape2d_t &shape) {
  contentType_ = contentType;
  reshape(shape);
}

matrix_t& Edge::getData() {
  return data_;
}

const matrix_t& Edge::getData() const {
  return data_;
}

matrix_t& Edge::getGradient() {
  return gradient_;
}

const matrix_t& Edge::getGradient() const {
  return gradient_;
}

content_t Edge::getContentType() const {
  return contentType_;
}

const shape2d_t& Edge::shape() const {
  return shape_;
}

void Edge::reshape(const shape2d_t& shape) {
  data_.reshape(shape);
  gradient_.reshape(shape);
  shape_ = shape;
}

}