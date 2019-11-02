#pragma once

#include <limits>
#include <random>
#include <type_traits>

namespace borjomi {

class RandomGenerator {
 
 private:
  RandomGenerator() : generator(1) {}
  std::mt19937 generator;

 public:
  static RandomGenerator &getInstance() {
    static RandomGenerator instance;
    return instance;
  }

  std::mt19937& getGenerator() {
    return generator;
  }

  void setSeed(size_t seed) {
    generator.seed(seed);
  }
};

float uniformRand(float min, float max) {
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(RandomGenerator::getInstance().getGenerator());
}

float gaussianRand(float mean, float sigma) {
  std::normal_distribution<float> distribution(mean, sigma);
  return distribution(RandomGenerator::getInstance().getGenerator());
}

void setRandomSeed(size_t seed) {
  RandomGenerator::getInstance().setSeed(seed);
}

void uniformRand(float* array, float min, float max, size_t size) {
  for (size_t idx = 0; idx < size; idx++) {
    array[idx] = uniformRand(min, max);
  }
}

void gaussianRand(float* array, float mean, float sigma, size_t size) {
  for (size_t idx = 0; idx < size; idx++) {
    array[size] = gaussianRand(mean, sigma);
  }
}

}