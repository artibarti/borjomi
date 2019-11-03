/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <cstring>

#include "borjomi/borjomi.h"

using namespace borjomi;

#define BATCH_SIZE 16
#define NUMBER_OF_EPOCHS 2000

void createTrainingData(matrix_t& x, matrix_t& sinx) {

  float from = -3.1416f;
  float to = 3.1416f;
  float step = 0.2f;
  size_t sampleCount = (to - from) / step + 1;
  x.reshape(sampleCount, 1);
  sinx.reshape(sampleCount, 1);

  for (float num = from, sampleIdx = 0; num <= to; num += step, sampleIdx++) {
    x.at(sampleIdx, 0) = num;
    sinx.at(sampleIdx, 0) = sinf(num);
  }
}

void train(Network& net, matrix_t& x, matrix_t& sinx) {

  Adamax opt;
  int epochIdx = 0;

  auto onEnumerateEpoch = [&]() {  
    if (epochIdx%100 == 0 && epochIdx != 0) {
      float loss = net.getLoss<LossFunctionType::Mse>(x, sinx);
      std::cout << "epoch = " << epochIdx << "/"
        << NUMBER_OF_EPOCHS << "\tloss = " << loss << std::endl;
    }
    epochIdx++;
  };

  std::cout << std::endl << "Learning the sinus function with 2000 epochs..." << std::endl;
  net.fit<LossFunctionType::Mse>(opt, x, sinx, BATCH_SIZE, NUMBER_OF_EPOCHS, []() {}, onEnumerateEpoch);
}

void test(Network& net) {

  std::cout << std::endl << "Training finished, now computing prediction results:" << std::endl;
  float fMaxError = 0.f;
  for (float x = -3.1416; x < 3.1416; x += 0.2) {

    matrix_t xv(1,1, x);
    float fPredicted = net.predict(xv).at(0,0);
    float fDesired = sinf(x);

    std::cout << "x = " << std::fixed << std::setprecision(5) << x << "\tsin(x) = " << fDesired
      << "\tpredicted = " << fPredicted << std::endl;

    float fError = fabs(fPredicted - fDesired);
    if (fMaxError < fError) {
      fMaxError = fError;
    }
  }
  std::cout << std::endl << "Max error = " << fMaxError << std::endl;
}

int main(int argc, char** argv) {

  Network net;
  net << FullyConnectedLayer(1, 10, true, engine_t::internal);
  net << TanhLayer();
  net << FullyConnectedLayer(10, 10, true, engine_t::internal);
  net << TanhLayer();
  net << FullyConnectedLayer(10, 1, true, engine_t::internal);

  matrix_t x;
  matrix_t sinx;

  createTrainingData(x, sinx);
  train(net, x, sinx);
  test(net);

  return 0;
}
