#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "borjomi/borjomi.h"

using namespace borjomi;

struct MnistHeader {
  uint32_t magicNumber;
  uint32_t numItems;
  uint32_t numRows;
  uint32_t numCols;
};

bool isLittleEndian() {
  static std::int32_t test = 1;
  return *reinterpret_cast<std::int8_t*>( &test ) == 1;
}

template<typename T>
T* reverseEndian(T* p) {
  std::reverse(reinterpret_cast<char*>(p),
    reinterpret_cast<char*>(p) + sizeof(T));
  return p;
}

void parseMnistHeader(std::ifstream& ifs, MnistHeader& header) {

  ifs.read(reinterpret_cast<char *>(&header.magicNumber), 4);
  ifs.read(reinterpret_cast<char *>(&header.numItems), 4);
  ifs.read(reinterpret_cast<char *>(&header.numRows), 4);
  ifs.read(reinterpret_cast<char *>(&header.numCols), 4);

  if (isLittleEndian()) {
    reverseEndian(&header.magicNumber);
    reverseEndian(&header.numItems);
    reverseEndian(&header.numRows);
    reverseEndian(&header.numCols);
  }

  if (header.magicNumber != 0x00000803 || header.numItems <= 0)
    throw std::runtime_error("MNIST label-file format error");
  if (ifs.fail() || ifs.bad()) {
    throw std::runtime_error("File error");
  }
}

void parseMnistLabels(const std::string& labelFile, std::vector<label_t>* labels) {

  std::ifstream ifs(labelFile.c_str(), std::ios::in | std::ios::binary);

  if (ifs.bad() || ifs.fail())
    throw std::invalid_argument("Failed to open file: " + labelFile);

  uint32_t magic_number, num_items;

  ifs.read(reinterpret_cast<char *>(&magic_number), 4);
  ifs.read(reinterpret_cast<char *>(&num_items), 4);

  if (isLittleEndian()) {
    reverseEndian(&magic_number);
    reverseEndian(&num_items);
  }

  if (magic_number != 0x00000801 || num_items <= 0) {
    throw std::runtime_error("MNIST label-file format error");
  }

  labels->resize(num_items);
  for (uint32_t i = 0; i < num_items; i++) {
    uint8_t label;
    ifs.read(reinterpret_cast<char *>(&label), 1);
    (*labels)[i] = static_cast<label_t>(label);
  }
}

void parseMnistImage(std::ifstream &ifs, const MnistHeader &header, float_t scaleMin,
  float_t scaleMax, int xPadding, int yPadding, std::vector<float>& dst) {
  
  const int width  = header.numCols + 2 * xPadding;
  const int height = header.numRows + 2 * yPadding;

  std::vector<uint8_t> image_vec(header.numRows * header.numCols);
  ifs.read(reinterpret_cast<char *>(&image_vec[0]), header.numRows * header.numCols);
  dst.resize(width * height, scaleMin);

  for (uint32_t y = 0; y < header.numRows; y++) {
    for (uint32_t x = 0; x < header.numCols; x++) {
      dst[width * (y + yPadding) + x + xPadding] =
        (image_vec[y * header.numCols + x] / float_t(255)) *
          (scaleMax - scaleMin) + scaleMin;
    }
  }
}

void parseMnistImages(const std::string &imageFile, matrix_t* images,
  float_t scaleMin, float_t scaleMax, int xPadding, int yPadding) {

  if (xPadding < 0 || yPadding < 0)
    throw std::invalid_argument("Padding size must not be negative");
  if (scaleMin >= scaleMax)
    throw std::invalid_argument("Scale_max must be greater than scale_min");

  std::ifstream ifs(imageFile.c_str(), std::ios::in | std::ios::binary);

  if (ifs.bad() || ifs.fail())
    throw std::runtime_error("Failed to open file: " + imageFile);

  MnistHeader header;

  parseMnistHeader(ifs, header);

  const size_t imgWidth  = header.numCols + 2 * xPadding;
  const size_t imgHeight = header.numRows + 2 * yPadding;
  size_t imageSize = imgWidth * imgHeight;
  images->reshape(header.numItems, imageSize);

  for (uint32_t itemIdx = 0; itemIdx < header.numItems; itemIdx++) {
    std::vector<float> image;
    parseMnistImage(ifs, header, scaleMin, scaleMax, xPadding, yPadding, image);
    for (size_t idx = 0; idx < imageSize; idx++) {
      (*images).at(itemIdx, idx) = image[idx];
    }
  }
}