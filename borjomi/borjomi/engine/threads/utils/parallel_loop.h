
#pragma once

#include <string>
#include <vector>
#include <future>
#include <thread>

namespace borjomi {
 namespace engine {
  namespace threads {

class Idx2D {

  public:

  Idx2D(size_t x, size_t y, size_t dimx, size_t dimy, size_t stepx, size_t stepy) {
    x_ = x;
    y_ = y;
    dimx_ = dimx;
    dimy_ = dimy;
    stepx_ = stepx;
    stepy_ = stepy;
  }

  Idx2D(const Idx2D& other) {
    x_ = other.x_;
    y_ = other.y_;
    dimx_ = other.dimx_;
    dimy_ = other.dimy_;
    stepx_ = other.stepx_;
    stepy_ = other.stepy_;
  }

  void operator= (const Idx2D& other) {
    x_ = other.x_;
    y_ = other.y_;
    dimx_ = other.dimx_;
    dimy_ = other.dimy_;
    stepx_ = other.stepx_;
    stepy_ = other.stepy_;
  }

  void operator++ () {
    y_ += stepy_;
    if (y_ >= dimy_) {
      x_ += stepx_;
      y_ = 0;
    }
  }

  Idx2D operator+ (size_t value) const {
    Idx2D result = Idx2D(x_, y_, dimx_, dimy_, stepx_, stepy_);
    for (size_t i = 0; i < value; i++) {
      ++result;
    }
    return result;
  }

  void operator+= (size_t value) {
    for (size_t i = 0; i < value; i++) {
      operator++();
    }
  }

  bool operator>= (const Idx2D& other) const {
    return x_ > other.x_ || (x_ == other.x_ && y_ >= other.y_);
  }

  bool operator<= (const Idx2D& other) const {
    return x_ < other.x_ || (x_ == other.x_ && y_ <= other.y_);
  }

  bool operator> (const Idx2D& other) const {
    return x_ > other.x_ || (x_ == other.x_ && y_ > other.y_);
  }

  bool operator< (const Idx2D& other) const {
    return x_ < other.x_ || (x_ == other.x_ && y_ < other.y_);
  }

  size_t getx() const {
    return x_;
  }

  size_t gety() const {
    return y_;
  }

  size_t getStepX() const {
    return stepx_;
  }

  size_t getStepY() const {
    return stepy_;
  }

  private:
  
  size_t x_;
  size_t y_;
  size_t dimx_;
  size_t dimy_;
  size_t stepx_;
  size_t stepy_;
};

template<typename Func>
void parallelized2DLoop_(const size_t dimx, const size_t dimy,
  size_t stepx, size_t stepy, const Func &func) {

  size_t numberOfThreads  = std::thread::hardware_concurrency();
  size_t numberOfElementsPerThread = (dimx * dimy) / numberOfThreads;
  if (numberOfElementsPerThread * numberOfThreads < dimx * dimy) {
    numberOfElementsPerThread++;
  }

  std::vector<std::future<void>> futures;

  Idx2D begin(0,0, dimx, dimy, stepx, stepy);
  Idx2D end(begin + (numberOfElementsPerThread - 1));
  Idx2D maxIdx(dimx - 1, dimy - 1, dimx, dimy, stepx, stepy);

  if (end > maxIdx) {
    end = maxIdx;
  }

  for (size_t threadIdx = 0; threadIdx < numberOfThreads; threadIdx++) {
    futures.push_back(std::move(std::async(std::launch::async, [begin, end, &func] {
      func(begin, end);
    })));

    begin += numberOfElementsPerThread;
    end += numberOfElementsPerThread;

    if (begin > maxIdx) {
      break;
    }
    if (end > maxIdx) {
      end = maxIdx;
    }
  }
  for (auto &future : futures) {
    future.wait();
  }
}

template<typename Func>
inline void parallelized2DLoop(size_t dimx, size_t dimy, size_t stepx, size_t stepy, Func func) {
  parallelized2DLoop_(dimx, dimy, stepx, stepy, [&](const Idx2D& begin, const Idx2D& end) {
    for (Idx2D idx = begin; idx <= end; ++idx) {
      func(idx.getx(), idx.gety());
    }
  });
}

  }
 }
}