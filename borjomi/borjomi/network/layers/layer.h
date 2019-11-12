#pragma once

#include <algorithm>
#include <vector>
#include <map>

#include "borjomi/util/util.h"
#include "borjomi/network/edge.h"

namespace borjomi {

class Layer {
 
 private:
  shape3d_t inputShape_;
  shape3d_t outputShape_;

  bool isTrainable_;
  bool isActivation_;

  engine_t engine_;
  std::map<std::string, edgeptr_t> edges;

 public:  
  virtual ~Layer() = default;

  Layer();
  Layer(const shape3d_t& inShape, const shape3d_t& outShape,
    bool isActivation, bool isTrainable, engine_t engine = engine_t::internal);

  Layer(const Layer&) = default;
  Layer &operator=(const Layer&) = default;

  Layer(Layer&&) = default;
  Layer &operator=(Layer&&) = default;

  void setEngine(engine_t engine);
  engine_t getEngine() const;

  size_t getInDataSize() const;
  size_t getOutDataSize() const;

  bool isActivation() const;
  bool isTrainable() const;

  virtual std::pair<float, float> getOutValueRange() const;

  virtual std::string getLayerType() const = 0;

  shape3d_t getInputShape() const;
  shape3d_t getOutputShape() const;
  void setInputShape(const shape3d_t& shape);
  void setOutputShape(const shape3d_t& shape);

  virtual void forwardPropagation() = 0;
  virtual void backPropagation() = 0;

  void forward();
  void backward();

  virtual void initialize() {}
  virtual void setBatchSize(size_t batchSize = 1) = 0;

  void clearGradients();
  void updateWeight(Optimizer *o);

  void registerEdge(std::string name, content_t contentType);
  void reshapeEdge(std::string name, const shape2d_t& shape);

  edgeptr_t getEdge(std::string name);
  void setEdge(std::string name, edgeptr_t edge);

  matrix_t& getEdgeData(std::string edgeName);
  const matrix_t& getEdgeData(std::string edgeName) const;
  matrix_t& getEdgeGradient(std::string edgeName);
  const matrix_t& getEdgeGradient(std::string edgeName) const;

  void setEdgeData(std::string edgeName, const matrix_t& data);
  void setEdgeGradient(std::string edgeName, const matrix_t& gradient);
};

Layer::Layer() {}

Layer::Layer(const shape3d_t& inShape, const shape3d_t& outShape,
  bool isActivation, bool isTrainable, engine_t engine) {
  
  inputShape_ = inShape;
  outputShape_ = outShape;
  engine_ = engine;
  isTrainable_ = isTrainable;
  isActivation_ = isActivation;

  registerEdge("incomingEdge", content_t::data);
  registerEdge("outgoingEdge", content_t::data);
}

void Layer::setEngine(engine_t engine) {
  engine_ = engine;
}

engine_t Layer::getEngine() const {
  return engine_;
}

size_t Layer::getInDataSize() const {
  return inputShape_.size();
}

size_t Layer::getOutDataSize() const {
  return outputShape_.size();
}

bool Layer::isActivation() const {
  return isActivation_;
}

bool Layer::isTrainable() const {
  return isTrainable_;
}

std::pair<float, float> Layer::getOutValueRange() const {
  return {float{0.0}, float{1.0}};
}
  
void Layer::forward() {  
  clearGradients();
  forwardPropagation();
}

void Layer::backward() {
  backPropagation();
}

void Layer::clearGradients() {
  for (auto const& edge : edges) {
    matrix_t& grad = edge.second -> getGradient();
    for (size_t elementIdx = 0; elementIdx < grad.size(); elementIdx++) {
      grad.at(elementIdx) = float{0};
    }
  }
}

void Layer::updateWeight(Optimizer *optimizer) {
  for (auto const& entry : edges) {
    edgeptr_t edge = entry.second;
    if (edge->getContentType() == content_t::weight
      || edge->getContentType() == content_t::bias) {
      optimizer->update(edge->getGradient(), edge->getData());
    }
  }
}

shape3d_t Layer::getInputShape() const {
  return inputShape_;
}

shape3d_t Layer::getOutputShape() const {
  return outputShape_;
}

void Layer::setInputShape(const shape3d_t& shape) {
  inputShape_ = shape;
}

void Layer::setOutputShape(const shape3d_t& shape) {
  outputShape_ = shape;
}

void Layer::registerEdge(std::string name, content_t contentType) {
  edges[name] = std::make_shared<Edge>(contentType);
}

void Layer::reshapeEdge(std::string name, const shape2d_t& shape) {
  edges[name] -> reshape(shape);
}

edgeptr_t Layer::getEdge(std::string name) {
  return edges[name];
}

void Layer::setEdge(std::string name, edgeptr_t edge) {
  edges[name] = edge;
}

matrix_t& Layer::getEdgeData(std::string edgeName) {
  return edges.at(edgeName) -> getData();
}

const matrix_t& Layer::getEdgeData(std::string edgeName) const {
  return edges.at(edgeName) -> getData();
}

matrix_t& Layer::getEdgeGradient(std::string edgeName) {
  return edges.at(edgeName) -> getGradient();
}

const matrix_t& Layer::getEdgeGradient(std::string edgeName) const {
  return edges.at(edgeName) -> getGradient();
}

void Layer::setEdgeData(std::string edgeName, const matrix_t& data) {
  matrix_t& dest = getEdgeData(edgeName);
  if (dest.shape() != data.shape()) {
    throw BorjomiRuntimeException("Invalid shape");
  }
  dest = data;
}

void Layer::setEdgeGradient(std::string edgeName, const matrix_t& gradient) {
  matrix_t& dest = getEdgeGradient(edgeName);
  if (dest.shape() != gradient.shape()) {
    throw BorjomiRuntimeException("Invalid shape");
  }
  dest = gradient;
}

}