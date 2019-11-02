#pragma once

#include "borjomi/types/types.h"

#include <iostream>
#include <chrono>
#include <string>
#include <ratio>
#include <map>

namespace borjomi {

enum class DurationUnit {
  nanosec, microsec, millisec, sec
};

class Timer {

 private:
  std::chrono::high_resolution_clock::time_point start_, end_;

 public:
  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void stop() {    
    end_ = std::chrono::high_resolution_clock::now();
  }

  float getEllapsedTime(DurationUnit unit) {
    if (unit == DurationUnit::nanosec) {
      return std::chrono::duration<float, std::nano>(end_ - start_).count();
    } else if (unit == DurationUnit::microsec) {
      return std::chrono::duration<float, std::micro>(end_ - start_).count();
    } else if (unit == DurationUnit::millisec) {
      return std::chrono::duration<float, std::milli>(end_ - start_).count();
    } else if (unit == DurationUnit::sec) {
      return std::chrono::duration<float, std::ratio<1,1>>(end_ - start_).count();
    } else {
      throw new BorjomiRuntimeException("Duration unit is not supported");
    }
  }
};

}