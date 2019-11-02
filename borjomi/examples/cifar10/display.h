/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#pragma once

#include <iostream>
#include <string>

namespace borjomi {

class ProgressDisplay {
 
 private:
  size_t count_, expectedCount_, nextTicCount_, tic_;
  
  void displayTic() {
  
    size_t ticsNeeded = static_cast<size_t>((static_cast<double>(count_) / expectedCount_) * 50.0);  

    do {
      std::cout << '*' << std::flush;
    } while (++tic_ < ticsNeeded);
    
    nextTicCount_ = static_cast<size_t>((tic_ / 50.0) * expectedCount_);
    if (count_ == expectedCount_) {
      if (tic_ < 51) {
        std::cout << '*';
      }
      std::cout << std::endl;
    }
  }

  ProgressDisplay& operator=(const ProgressDisplay&) = delete;

 public:
  ProgressDisplay(size_t expectedCount) {
    restart(expectedCount);
  }

  void restart(size_t expectedCount) {
    count_ = nextTicCount_ = tic_ = 0;
    expectedCount_ = expectedCount;

    std::cout << std::endl
      << "0%   10   20   30   40   50   60   70   80   90   100%" << std::endl
      << "|----|----|----|----|----|----|----|----|----|----|" << std::endl;
    
    if (!expectedCount_) {
      expectedCount_ = 1;
    }
  }

  size_t operator+=(size_t increment) {
    if ((count_ += increment) >= nextTicCount_) {
      displayTic();
    }
    return count_;
  }

  size_t operator++() {
    return operator+=(1);
  }
  
  size_t count() const {
    return count_;
  }
  
  size_t expectedCount() const {
    return expectedCount_;
  }
};

}