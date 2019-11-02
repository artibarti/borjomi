#pragma once

#include <iostream>
#include <string>

#include "borjomi/types/types.h"

namespace borjomi {

template<typename T>
void log(const Matrix<T>& matrix, std::string tag = "unnamed matrix", bool logFull = false) {
  std::cout << std::endl;
  std::cout << tag << " (matrix) (" << matrix.shape() << ")" << ":" << std::endl;
  for (unsigned rowIndex = 0; rowIndex < matrix.rows(); rowIndex++) {
    for (unsigned colIndex = 0; colIndex < matrix.cols(); colIndex++) {
      std::cout << " " << matrix.at(rowIndex, colIndex);
      if (colIndex > 10 && !logFull) {
        break;
      }
    }
    if (rowIndex > 10 && !logFull) {
      break;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

}