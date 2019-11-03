#pragma once

namespace borjomi {

struct Result {

 private:
  size_t numberOfSuccessfulPredictions;
  size_t numberOfTotalPredictions;

 public:   
  Result() : numberOfSuccessfulPredictions(0), numberOfTotalPredictions(0) {}

  size_t getNumberOfTotalPredictions() const {
    return numberOfTotalPredictions;
  }

  size_t getNumberOfSuccessfulPredictions() const {
    return numberOfSuccessfulPredictions;
  }

  float getAccuracy() const {
    return float(numberOfSuccessfulPredictions * 100.0 / numberOfTotalPredictions);
  }

  void addPredictionResult(bool success) {
    numberOfTotalPredictions++;
    if (success) {
      numberOfSuccessfulPredictions++;
    }
  }
};

}