#pragma once

#include <memory>
#include <vector>

#include "borjomi/network/layers/layer.h"
#include "borjomi/util/util.h"

namespace borjomi {

class LayerSequence {
 
 private:
  std::vector<std::shared_ptr<Layer>> layers;

  void setInputData(const matrix_t& input);
  void setOutputGradient(const matrix_t& grad);

  void setBatchSize(size_t batchSize);
  void connect(Layer* first, Layer* second);
  bool checkConnectivity(Layer* head, Layer* tail);

 public:
  Layer* getLayer(size_t index);
  const Layer* getLayer(size_t index) const;
  Layer* operator[](size_t index);
  const Layer* operator[](size_t index) const;

  void initialize();
  void updateWeights(Optimizer* optimizer);

  size_t getSize() const;
  size_t getInDataSize() const;
  size_t getOutDataSize() const;

  matrix_t forward(const matrix_t& input);
  void backward(const matrix_t& grads);

  template<typename T>
  void addLayer(T&& layer);
};

Layer* LayerSequence::getLayer(size_t index) {
  return layers[index].get();
}

const Layer* LayerSequence::getLayer(size_t index) const {
  return layers[index].get();
}

void LayerSequence::updateWeights(Optimizer* optimizer) {
  for (size_t idx = 0; idx < layers.size(); idx++) {
    getLayer(idx) -> updateWeight(optimizer);
  }    
}

void LayerSequence::initialize() {
  for (size_t idx = 0; idx < layers.size(); idx++) {
    getLayer(idx) -> initialize();    
  }
}

void LayerSequence::setBatchSize(size_t batchSize) {
  for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
    getLayer(layerIdx) -> setBatchSize(batchSize);
  }
}

size_t LayerSequence::getSize() const {
  return layers.size();
}

Layer* LayerSequence::operator[](size_t index) {
  return getLayer(index);
}
  
const Layer* LayerSequence::operator[](size_t index) const {
  return getLayer(index);
}
  
size_t LayerSequence::getInDataSize() const {
  return getLayer(0) -> getInDataSize();
}
  
size_t LayerSequence::getOutDataSize() const {
  return getLayer(layers.size() - 1) -> getOutDataSize();
}

matrix_t LayerSequence::forward(const matrix_t& input) {

  setBatchSize(input.rows());
  setInputData(input);

  for (size_t idx = 0; idx < layers.size(); idx++) {
    getLayer(idx) -> forward();
  }
  return getLayer(layers.size() - 1) -> getEdgeData("outgoingEdge");
}

void LayerSequence::backward(const matrix_t& grad) {

  setOutputGradient(grad);
  for (int idx = layers.size() - 1; idx >= 0; idx--) {
    getLayer(idx) -> backward();
  }
}

void LayerSequence::setInputData(const matrix_t& input) {
  matrix_t& layerInput = getLayer(0) -> getEdgeData("incomingEdge");
  if (layerInput.shape() != input.shape()) {
    throw BorjomiRuntimeException("Input data is invalid");
  }
  layerInput = input;
}

void LayerSequence::setOutputGradient(const matrix_t& grad) {
  matrix_t& outGrad = getLayer(layers.size() - 1) -> getEdgeGradient("outgoingEdge");
  if (outGrad.shape() != grad.shape()) {
    throw BorjomiRuntimeException("Gradient data is invalid");
  }
  outGrad = grad;
}


void LayerSequence::connect(Layer* first, Layer* second) {
  auto outShape = first->getOutputShape();
  auto inShape  = second->getInputShape();
  if (outShape.size() != inShape.size()) {
    throw BorjomiRuntimeException("Impossible to connect layers due to dimension mismatch");
  }
  second->setEdge("incomingEdge", first->getEdge("outgoingEdge"));
}

template<typename T>
void LayerSequence::addLayer(T&& layer) {

  layers.push_back(std::make_shared<typename std::remove_reference<T>::type>(std::forward<T>(layer)));

  if (layers.back() -> isActivation()) {
    if (layers.size() == 1) {
      throw new BorjomiRuntimeException("An activation layer should not be the first one in the network");
    } else {
      getLayer(layers.size() - 2) -> getOutputShape();
      layers.back() -> setInputShape(getLayer(layers.size() - 2) -> getOutputShape());
      layers.back() -> setOutputShape(getLayer(layers.size() - 2) -> getOutputShape());
    }
  }

  if (layers.size() != 1) {
    auto first = getLayer(layers.size() - 2);
    auto second = getLayer(layers.size() - 1);

    connect(first, second);
    if (!checkConnectivity(first, second)) {
      throw BorjomiRuntimeException("Failed to connect layers");
    }
  }
}

bool LayerSequence::checkConnectivity(Layer* head, Layer* tail) {
  return head -> getEdge("outgoingEdge") == tail -> getEdge("incomingEdge");
}

}