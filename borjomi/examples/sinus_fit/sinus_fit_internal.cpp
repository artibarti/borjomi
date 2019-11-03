/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <cstring>

#include "borjomi/borjomi.h"

using namespace borjomi;

void createTrainingData(matrix_t& x, matrix_t& sinx) {

  float from = -3.1416f;
  float to = 3.1416f;
  float step = 0.2f;
  size_t sampleCount = (to - from) / step + 1;
  x.reshape(sampleCount, 1);
  sinx.reshape(sampleCount, 1);

  unsigned index = 0;
  for (float num = from; num < to; num += step) {
    float vx = {num};
    float vsinx = {sinf(num)};
    x.at(index, 0) = vx;
    sinx.at(index, 0) = vsinx;
    index++;
  }
}

void train(Network& net, matrix_t& x, matrix_t& sinx) {

  unsigned BATCH_SIZE = 16;
  unsigned NUMBER_OF_EPOCHS = 2000;
  Adamax opt;
  
  int iEpoch = 0;
  auto onEnumerateEpoch = [&]() {  
    iEpoch++;
    if (iEpoch % 100) return;
    float loss = net.getLoss<LossFunctionType::Mse>(x, sinx);
    std::cout << "epoch = " << iEpoch << "/" << NUMBER_OF_EPOCHS << " loss = " << loss << std::endl;
  };

  std::cout << "learning the sinus function with 2000 epochs:" << std::endl;
  net.fit<LossFunctionType::Mse>(opt, x, sinx, BATCH_SIZE, NUMBER_OF_EPOCHS, []() {}, onEnumerateEpoch);
}

void test(Network& net) {

  std::cout << "Training finished, now computing prediction results:" << std::endl;
  float fMaxError = 0.f;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {

    matrix_t xv(1,1);
    xv.at(0, 0) = x;
    float fPredicted = net.predict(xv).at(0, 0);
    float fDesired = sinf(x);

    std::cout << "x = " << x << " sin(x) = " << fDesired << "   predicted = " << fPredicted << std::endl;

    float fError = fabs(fPredicted - fDesired);
    if (fMaxError < fError) {
      fMaxError = fError;
    }
  }
  std::cout << std::endl << "Max error = " << fMaxError << std::endl;
}

int main() {

  Network net;
  net << FullyConnectedLayer(1, 10, true, borjomi::engine_t::internal);
  net << TanhLayer();
  net << FullyConnectedLayer(10, 10, true, borjomi::engine_t::internal);
  net << TanhLayer();
  net << FullyConnectedLayer(10, 1, true, borjomi::engine_t::internal);

  matrix_t x;
  matrix_t sinx;

  createTrainingData(x, sinx);
  train(net, x, sinx);
  test(net);

  return 0;
}
