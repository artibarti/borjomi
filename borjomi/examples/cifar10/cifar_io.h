/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "borjomi/borjomi.h"

using namespace borjomi;

#define CIFAR10_IMAGE_DEPTH (3)
#define CIFAR10_IMAGE_WIDTH (32)
#define CIFAR10_IMAGE_HEIGHT (32)
#define CIFAR10_IMAGE_AREA (CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT)
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_AREA * CIFAR10_IMAGE_DEPTH)
#define SCALE_MIN -1.0
#define SCALE_MAX 1.0
#define PADDING_X 0
#define PADDING_Y 0

void readCifarTrainData(const std::string& baseDir, matrix_t& images, std::vector<label_t>& labels) {

  images.reshape(50000, CIFAR10_IMAGE_SIZE);

  for (unsigned i = 1; i <=5; i++) {
    
    std::string path = baseDir + "/data_batch_" + std::to_string(i) + ".bin";
    std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail() || ifs.bad()) {
      throw BorjomiRuntimeException("readCifarTrainData: failed to open file:" + path);
    }
  
    uint8_t label;
    std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);
    int imageIndex = 0;

    while (ifs.read(reinterpret_cast<char *>(&label), 1)) {
    
      std::vector<float> img;
      if (!ifs.read(reinterpret_cast<char *>(&buf[0]), CIFAR10_IMAGE_SIZE)) {
        break;
      }

      std::transform(buf.begin(), buf.end(), std::back_inserter(img), [=](unsigned char c) {
        return SCALE_MIN + (SCALE_MAX - SCALE_MIN) * c / 255;
      });

      images.setRow( (i-1) * 10000 + imageIndex , img);
      labels.push_back(label);
      imageIndex++;
    }
  }
}

void readCifarTestData(const std::string& baseDir, matrix_t& images, std::vector<label_t>& labels) {

  images.reshape(10000, CIFAR10_IMAGE_SIZE);
    
  std::string path = baseDir + "/test_batch.bin";
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (ifs.fail() || ifs.bad()) {
    throw BorjomiRuntimeException("readCifarTestData: failed to open file:" + path);
  }
  
  uint8_t label;  
  std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);
  int imageIndex = 0;

  while (ifs.read(reinterpret_cast<char *>(&label), 1)) {
    
    std::vector<float> img;
    if (!ifs.read(reinterpret_cast<char *>(&buf[0]), CIFAR10_IMAGE_SIZE)) {
      break;
    }

    std::transform(buf.begin(), buf.end(), std::back_inserter(img), [=](unsigned char c) {
      return SCALE_MIN + (SCALE_MAX - SCALE_MIN) * c / 255;
    });

    images.setRow( imageIndex , img);
    labels.push_back(label);
    imageIndex++;
  }
}