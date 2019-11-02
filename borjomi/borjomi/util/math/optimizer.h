/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <map>
#include <cmath>

#include "borjomi/types/types.h"

namespace borjomi {

struct Optimizer {
  
  Optimizer() = default;
  Optimizer(const Optimizer &) = default;
  Optimizer(Optimizer &&) = default;

  virtual ~Optimizer() = default;

  Optimizer& operator=(const Optimizer &) = default;
  Optimizer& operator=(Optimizer &&) = default;
  
  virtual void update(const matrix_t& dW, matrix_t& W) = 0;
  virtual void reset() {}
};

template<int N>
struct StatefulOptimizer : public Optimizer {
  
 private:
  std::map<const matrix_t*, matrix_t> E_[N];

 public: 
  void reset() override {
    for (auto &e : E_) {
      e.clear();
    }
  }

 protected:
  template <int Index>
  matrix_t& get(const matrix_t& key) {
    if (Index >= N) {
      throw BorjomiRuntimeException("StatefulOptimizer: Index out of bounds");
    }
    if (E_[Index].find(&key) == E_[Index].end()) {
      E_[Index][&key] = matrix_t(key.shape());
      for (size_t index = 0; index < key.size(); index++) {
        E_[Index][&key].at(index) = 0;
      }
    }
    return E_[Index][&key];
  }  
};

struct Adagrad : public StatefulOptimizer<1> {
  
 public:
  Adagrad() : learningRate(float(0.01)), eps(float(1e-8)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    matrix_t& g = get<0>(W);
    for(size_t i = 0; i < W.size(); i++) {
      g.at(i) += dW.at(i) * dW.at(i);
      W.at(i) -= learningRate * dW.at(i) / (std::sqrt(g.at(i)) + eps);
    }
  }

  float learningRate;
 private:
  float eps;
};

struct RMSprop : public StatefulOptimizer<1> {
  
 public:
  RMSprop() : learningRate(float(0.0001)),
    decayTerm(float(0.99)), eps(float(1e-8)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    
    matrix_t& g = get<0>(W);
    for(size_t i = 0; i < W.size(); i++) {
      g.at(i) = decayTerm * g.at(i) + (1 - decayTerm) * dW.at(i) * dW.at(i);
      W.at(i) -= learningRate * dW.at(i) / std::sqrt(g.at(i) + eps);
    }
  }

  float learningRate;
  float decayTerm;

 private:
  float eps;
};

struct Adam : public StatefulOptimizer<2> {
  
 public:
  Adam() : alpha(float(0.001)), b1(float(0.9)),
    b2(float(0.999)), b1_t(float(0.9)),
    b2_t(float(0.999)), eps(float(1e-8)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    
    matrix_t& mt = get<0>(W);
    matrix_t& vt = get<1>(W);

    for(size_t i = 0; i < W.size(); i++) {
      mt.at(i) = b1 * mt.at(i) + (float(1) - b1) * dW.at(i);
      vt.at(i) = b2 * vt.at(i) + (float(1) - b2) * dW.at(i) * dW.at(i);
      W.at(i) -= alpha * (mt.at(i) / (float(1) - b1_t)) / std::sqrt((vt.at(i) / (float(1) - b2_t)) + eps);
    }

    b1_t *= b1;
    b2_t *= b2;
  }

  float alpha;
  float b1;     
  float b2;     
  float b1_t;
  float b2_t;

 private:
  float eps;
};

struct Adamax : public StatefulOptimizer<2> {
  
 public:  
  Adamax() : alpha(float(0.002)),
    b1(float(0.9)), b2(float(0.999)),
    b1_t(b1), eps(float(1e-8)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    matrix_t& mt = get<0>(W);
    matrix_t& ut = get<1>(W);

    for(size_t i = 0; i < W.size(); i++) {
      mt.at(i) = b1 * mt.at(i) + (float(1) - b1) * dW.at(i);
      ut.at(i) = std::max(b2 * ut.at(i), std::abs(dW.at(i)));
      W.at(i) -= (alpha / (1.0 - b1_t)) * (mt.at(i) / (ut.at(i) + eps));
    }

    b1_t *= b1;
  }

  float alpha;
  float b1;
  float b2;
  float b1_t;

 private:
  float eps;
};

struct GradientDescent : public Optimizer {
  GradientDescent() : learningRate(float(0.01)), weightDecay(float(0)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    for(size_t i = 0; i < W.size(); i++) {
      W.at(i) = W.at(i) - learningRate * (dW.at(i) + weightDecay * W.at(i));
    }
  }
  float learningRate;
  float weightDecay;
};

struct Momentum : public StatefulOptimizer<1> {
 
 public:
  Momentum() : learningRate(float(0.01)), weightDecay(float(0)), momentum(float(0.9)) {}

  void update(const matrix_t& dW, matrix_t& W) {  
    matrix_t& dWprev = get<0>(W);
    for(size_t i = 0; i < W.size(); i++) {
      float V = momentum * dWprev.at(i) - learningRate * (dW.at(i) + W.at(i) * weightDecay);
      W.at(i) += V;
      dWprev.at(i) = V;
    }
  }
  float learningRate;
  float weightDecay;
  float momentum;
};

struct NesterovMomentum : public StatefulOptimizer<1> {
 public:
  NesterovMomentum() : learningRate(float(0.01)), weightDecay(float(0)), momentum(float(0.9)) {}

  void update(const matrix_t& dW, matrix_t& W) {
    matrix_t& dWprev = get<0>(W);
    for(size_t i = 0; i < W.size(); i++) {
      float V = momentum * dWprev.at(i) - learningRate * (dW.at(i) + W.at(i) * weightDecay);
      W.at(i) += (-momentum) * dWprev.at(i) + (1 + momentum) * V;
      dWprev.at(i) = V;
    }
  }
  float learningRate;
  float weightDecay;
  float momentum;
};

}