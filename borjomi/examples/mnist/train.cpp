/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "borjomi/borjomi.h"
#include "mnist_parser.h"
#include "display.h"

using namespace borjomi;

static void constructNet(Network& network, engine_t engine) {

  // connection table [Y.Lecun, 1998 Table.1]
  #define O true
  #define X false
  // clang-format off
  static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
  };
  // clang-format on
  #undef O
  #undef X

  using fc = FullyConnectedLayer;
  using conv = ConvolutionalLayer;
  using ave_pool = AveragePoolingLayer;
  using tanh = TanhLayer;

  // using tiny_dnn::core::connection_table;


    /*
  nn << conv(32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
     << tanh()
     << conv(14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
             connection_table(tbl, 6, 16),
             padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
     << tanh()
     << conv(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out 
        padding::valid, true, 1, 1, 1, 1, backend_type)
     << tanh()
     << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
     << tanh();
    */

  network << conv(32, 32, 1, 5, 5, 6, padding::valid, true, engine)
        << tanh()
        << ave_pool(28, 28, 6, 2)
        << tanh()
        // << conv(14, 14, 5, 6, 16, connection_table(tbl, 6, 16), padding::valid, true, engine)
        << conv(14, 14, 6, 5, 5, 16, padding::valid, true, engine)
        << tanh()
        << ave_pool(10, 10, 16, 2)
        << tanh()
        << conv(5, 5, 16, 5, 5, 120, padding::valid, true, engine)
        << tanh()
        << fc(120, 10, true, engine)
        << tanh();
}

static void train(const std::string &dataDirPath, double learningRate,
    const int numberOfEpochs, const int batchSize, engine_t engine) {

  // specify loss-function and learning strategy
  Network network;
  Adagrad optimizer;

  constructNet(network, engine);

  std::cout << "Loading models..." << std::endl;

  std::vector<label_t> train_labels, test_labels;
  matrix_t train_images, test_images;

  parseMnistLabels(dataDirPath + "/train-labels-idx1-ubyte", &train_labels);
  parseMnistImages(dataDirPath + "/train-images-idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
  parseMnistLabels(dataDirPath + "/t10k-labels-idx1-ubyte", &test_labels);
  parseMnistImages(dataDirPath + "/t10k-images-idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

  std::cout << "start training" << std::endl;

  ProgressDisplay disp(train_images.rows());
  Timer timer;

  optimizer.learningRate *=
    std::min(float_t(4), static_cast<float_t>(sqrt(batchSize) * learningRate));

  int epoch = 1;
  auto on_enumerate_epoch = [&]() {
    timer.stop();
    std::cout << "Epoch " << epoch << "/" << numberOfEpochs << " finished. "
        << timer.getEllapsedTime(DurationUnit::nanosec) << "s elapsed." << std::endl;
    ++epoch;
    Result res = network.test(test_images, test_labels);
    std::cout << res.getNumberOfSuccessfulPredictions() << "/" << res.getNumberOfTotalPredictions() << std::endl;

    disp.restart(train_images.rows());
    timer.start();
  };

  auto on_enumerate_minibatch = [&]() {
      disp += batchSize;
  };

  network.fit<LossFunctionType::Mse>(optimizer, train_images, train_labels,
    batchSize, numberOfEpochs, on_enumerate_minibatch, on_enumerate_epoch);

  std::cout << "Training has finished" << std::endl;
  std::cout << "Testing..." << std::endl;

  Result res = network.test(test_images, test_labels);
  std::cout << res.getNumberOfSuccessfulPredictions() << "/" << res.getNumberOfTotalPredictions() << std::endl;
  // network.save("LeNet-model");
}

static engine_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<engine_t>(i);
    }
  }
  return engine_t::internal;
}

static void showUsage(const char* argv0) {
  std::cout
    << "Usage: " << argv0
    << " --data_path path_to_dataset_folder"
    << " --learning_rate 1"
    << " --epochs 30"
    << " --minibatch_size 16"
    << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate = 1;
  int epochs = 30;
  std::string data_path = "";
  int minibatch_size = 16;
  engine_t engine = engine_t::internal;

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      showUsage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      engine = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\"" << std::endl;
      showUsage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    showUsage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr << "Invalid learning rate. The learning rate must be greater than 0." << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be greater than 0." << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 60000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0 and less than dataset size (60000)." << std::endl;
    return -1;
  }

  std::cout
    << "Running with the following parameters:" << std::endl
    << "Data path: " << data_path << std::endl
    << "Learning rate: " << learning_rate << std::endl
    << "Minibatch size: " << minibatch_size << std::endl
    << "Number of epochs: " << epochs << std::endl
    << "Backend type: " << engine << std::endl
    << std::endl;

  train(data_path, learning_rate, epochs, minibatch_size, engine);
  return 0;
}
