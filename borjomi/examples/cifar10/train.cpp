/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>

#include "borjomi/borjomi.h"
#include "cifar_io.h"
#include "display.h"

using namespace borjomi;

void constructNetwork(Network& net) {

  using conv = ConvolutionalLayer;
  using convv2 = ConvolutionalLayerV2;
  using pool = MaxPoolingLayer;
  using fc = FullyConnectedLayer;
  using relu = ReluLayer;
  using softmax = SoftmaxLayer;

  // net << conv(32, 32, 3, 5, 5, 32, padding::same, true, engine_t::avx);
  net << convv2(32, 32, 3, 5, 32, padding::same, true, engine_t::internal);
  net << pool(32, 32, 32, 2, engine_t::threads);
  net << relu();
  net << convv2(16, 16, 32, 5, 32, padding::same, true, engine_t::internal);
  // net << conv(16, 16, 32, 5, 5, 32, padding::same, true, engine_t::avx);
  net << pool(16, 16, 32, 2, engine_t::threads);
  net << relu();
  net << conv(8, 8, 32, 5, 5, 64, padding::same, true, engine_t::avx);
  // net << convv2(8, 8, 32, 5, 64, padding::same, true, engine_t::internal);
  net << pool(8, 8, 64, 2, engine_t::threads);
  net << relu();
  net << fc(1024, 64, true, engine_t::avx);
  net << relu();
  net << fc(64, 10, true, engine_t::avx);
  net << softmax();
}

void train(std::string dataDirPath) {

  double learningRate = 0.01;
  int minibatchSize = 10;
  int epochs = 30;

  borjomi::Network net;
  borjomi::Adam optimizer;
  constructNetwork(net);

  std::vector<label_t> trainLabels, testLabels;
  matrix_t trainImages, testImages;

  std::cout << "Loading training images..." << std::endl;
  readCifarTrainData(dataDirPath, trainImages, trainLabels);

  std::cout << "Loading test images..." << std::endl;
  readCifarTestData(dataDirPath, testImages, testLabels);

  std::cout << "Start learning" << std::endl;
  ProgressDisplay disp(trainImages.rows());
  Timer timer;
  timer.start();

  optimizer.alpha *= static_cast<float>(sqrt(minibatchSize) * learningRate);

  int epoch = 1;
  auto onEnumerateEpoch = [&]() {
    timer.stop();
    std::cout << "Epoch " << epoch << "/" << epochs << " finished. ";
    std::cout << timer.getEllapsedTime(DurationUnit::sec) << "s elapsed." << std::endl;
    borjomi::Result res = net.test(testImages, testLabels);
    std::cout << res.getNumberOfSuccessfulPredictions() << "/" << res.getNumberOfTotalPredictions() << std::endl;
    disp.restart(trainImages.rows());
    epoch++;
    timer.start();
  };

  auto onEnumerateMinibatch = [&]() { disp += minibatchSize; };

  net.fit<LossFunctionType::CrossEntropy>(optimizer, trainImages, trainLabels,
    minibatchSize, epochs, onEnumerateMinibatch, onEnumerateEpoch);
  
  std::cout << "Training finished" << std::endl;
  Result result = net.test(testImages, testLabels);
  
  std::cout << "Accuracy:" << result.getAccuracy() << "% ("
    << result.getNumberOfSuccessfulPredictions() << "/"
    << result.getNumberOfTotalPredictions() << ")" << std::endl;
}

int main(int argc, char **argv) {

  int epochs = 30;
  double learningRate = 0.01;
  std::string data_path = "";
  int minibatchSize = 10;

  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\"" << std::endl;
      return -1;
    }
  }

  if (data_path == "") {
    std::cerr << "Specifie train data path please..." << std::endl;
    return -1;
  }

  std::cout << "Running with the following parameters:" << std::endl
    << "Data path: " << data_path << std::endl
    << "Learning rate: " << learningRate << std::endl
    << "Minibatch size: " << minibatchSize << std::endl
    << "Number of epochs: " << epochs << std::endl
    << std::endl;
  
  try {
    train(data_path);
  } catch (borjomi::BorjomiRuntimeException &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}