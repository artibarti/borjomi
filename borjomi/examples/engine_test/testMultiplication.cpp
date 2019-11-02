
#include "borjomi/borjomi.h"
#include <cstdlib>
#include <iostream>

using namespace borjomi;

float generateRandomFloat() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 10.0);
}

std::string compare(const matrix_t& m1, const matrix_t& m2, float errorThreshold = 0.0) {
  if (m1.shape() != m2.shape()) {
    return "failed (different shapes)";
  }
  size_t errs = 0;

  for (size_t elementIdx = 0; elementIdx < m1.size(); elementIdx++) {
    if ( std::abs(m1.at(elementIdx) - m2.at(elementIdx)) > errorThreshold ) {
      errs++;
    }
  }
  if (errs) {
    std::string res = "failed (" + std::to_string(errs) + ")";
    return res;
  }
  return "passed";
}

void multiply(engine_t engine, float alpha, bool transposeLeft, matrix_t& left,
  bool transposeRight, matrix_t& right, float beta, matrix_t& result, float& ellapsedTime) {

  Timer timer;
  timer.start();
  multiply(engine, 1.0, transposeLeft, left, transposeRight, right, 0.0, result);
  timer.stop();
  ellapsedTime = timer.getEllapsedTime();
}

void testNonTransposed(matrix_t& left, matrix_t& right) {

  std::cout << std::endl << "Test transpose non" << std::endl;
  std::cout << "------------------------" << std::endl;
  
  float ellapsedTimeInternal = 0;
  float ellapsedTimeThreads = 0;
  float ellapsedTimeAvx = 0;
  float ellapsedTimeCuda = 0;

  matrix_t resultInternal(left.rows(), right.cols(), 0);
  matrix_t resultThreads(left.rows(), right.cols(), 0);
  matrix_t resultAvx(left.rows(), right.cols(), 0);
  matrix_t resultCuda(left.rows(), right.cols(), 0);

  multiply(engine_t::internal, 1.0, false, left, false, right, 0.0, resultInternal, ellapsedTimeInternal);
  multiply(engine_t::threads, 1.0, false, left, false, right, 0.0, resultThreads, ellapsedTimeThreads);
  multiply(engine_t::cuda, 1.0, false, left, false, right, 0.0, resultCuda, ellapsedTimeCuda);
  multiply(engine_t::avx, 1.0, false, left, false, right, 0.0, resultAvx, ellapsedTimeAvx);

  std::cout << "Internal version ellapsed time: " << ellapsedTimeInternal << " ns" << std::endl;
  std::cout << "Threaded version ellapsed time: " << ellapsedTimeThreads << " ns" << std::endl;
  std::cout << "AVX version elapsed time:       " << ellapsedTimeAvx << " ns" << std::endl;
  std::cout << "CUDA version ellapsed time:     " << ellapsedTimeCuda << " ns" << std::endl;

  std::cout << "------------------------" << std::endl;

  std::cout << "Check threaded version: " << compare(resultInternal, resultThreads) << std::endl;
  std::cout << "Check AVX version:      " << compare(resultInternal, resultAvx) << std::endl;
  std::cout << "Check CUDA version:     " << compare(resultInternal, resultCuda) << std::endl;
}

void testTransposeRight(matrix_t& left, matrix_t& right) {

  std::cout << std::endl << "Test transpose right" << std::endl;
  std::cout << "------------------------" << std::endl;

  float ellapsedTimeInternal = 0;
  float ellapsedTimeThreads = 0;
  float ellapsedTimeAvx = 0;
  float ellapsedTimeCuda = 0;

  matrix_t resultInternal(left.rows(), right.rows(), 0);
  matrix_t resultThreads(left.rows(), right.rows(), 0);
  matrix_t resultAvx(left.rows(), right.rows(), 0);
  matrix_t resultCuda(left.rows(), right.rows(), 0);

  multiply(engine_t::internal, 1.0, false, left, true, right, 0.0, resultInternal, ellapsedTimeInternal);
  multiply(engine_t::threads, 1.0, false, left, true, right, 0.0, resultThreads, ellapsedTimeThreads);
  multiply(engine_t::cuda, 1.0, false, left, true, right, 0.0, resultCuda, ellapsedTimeCuda);
  multiply(engine_t::avx, 1.0, false, left, true, right, 0.0, resultAvx, ellapsedTimeAvx);

  std::cout << "Internal version ellapsed time: " << ellapsedTimeInternal << " ns" << std::endl;
  std::cout << "Threaded version ellapsed time: " << ellapsedTimeThreads << " ns" << std::endl;
  std::cout << "AVX version elapsed time:       " << ellapsedTimeAvx << " ns" << std::endl;
  std::cout << "CUDA version ellapsed time:     " << ellapsedTimeCuda << " ns" << std::endl;

  std::cout << "------------------------" << std::endl;

  std::cout << "Check threaded version: " << compare(resultInternal, resultThreads) << std::endl;
  std::cout << "Check AVX version:      " << compare(resultInternal, resultAvx) << std::endl;
  std::cout << "Check CUDA version:     " << compare(resultInternal, resultCuda) << std::endl;
}

void testTransposeLeft(matrix_t& left, matrix_t& right) {

  std::cout << std::endl << "Test transpose left" << std::endl;
  std::cout << "------------------------" << std::endl;

  float ellapsedTimeInternal = 0;
  float ellapsedTimeThreads = 0;
  float ellapsedTimeAvx = 0;
  float ellapsedTimeCuda = 0;

  matrix_t resultInternal(left.cols(), right.cols(), 0);
  matrix_t resultThreads(left.cols(), right.cols(), 0);
  matrix_t resultAvx(left.cols(), right.cols(), 0);
  matrix_t resultCuda(left.cols(), right.cols(), 0);

  multiply(engine_t::internal, 1.0, true, left, false, right, 0.0, resultInternal, ellapsedTimeInternal);
  multiply(engine_t::threads, 1.0, true, left, false, right, 0.0, resultThreads, ellapsedTimeThreads);
  multiply(engine_t::cuda, 1.0, true, left, false, right, 0.0, resultCuda, ellapsedTimeCuda);
  multiply(engine_t::avx, 1.0, true, left, false, right, 0.0, resultAvx, ellapsedTimeAvx);

  std::cout << "Internal version ellapsed time: " << ellapsedTimeInternal << " ns" << std::endl;
  std::cout << "Threaded version ellapsed time: " << ellapsedTimeThreads << " ns" << std::endl;
  std::cout << "AVX version elapsed time:       " << ellapsedTimeAvx << " ns" << std::endl;
  std::cout << "CUDA version ellapsed time:     " << ellapsedTimeCuda << " ns" << std::endl;

  std::cout << "------------------------" << std::endl;

  std::cout << "Check threaded version: " << compare(resultInternal, resultThreads) << std::endl;
  std::cout << "Check AVX version:      " << compare(resultInternal, resultAvx) << std::endl;
  std::cout << "Check CUDA version:     " << compare(resultInternal, resultCuda) << std::endl;
}

int main(int argc, char** argv) {

  std::cout << "---------------------" << std::endl;
  std::cout << "Multiplication test" << std::endl << std::endl;

  shape2d_t shapes[6][2] = {
    {shape2d_t(10, 10),  shape2d_t(10, 8000)},
    {shape2d_t(42, 120), shape2d_t(120, 8200)},
    {shape2d_t(10, 50),  shape2d_t(50, 1200)},
    {shape2d_t(230, 10), shape2d_t(10, 82200)},
    {shape2d_t(2, 230),  shape2d_t(230, 333)},
    {shape2d_t(10, 64),  shape2d_t(64, 10)},
  };

  for (size_t idx = 0; idx < 6; idx++) {

    std::cout << std::endl << std::endl << "---------------------" << std::endl;
    std::cout << "Testing for " << shapes[idx][0] << " x " << shapes[idx][1] << std::endl;
    std::cout << "---------------------" << std::endl;

    matrix_t left(shapes[idx][0]);
    matrix_t right(shapes[idx][1]);

    for (size_t elementIdx = 0; elementIdx < left.size(); elementIdx++) {
      left.at(elementIdx) = generateRandomFloat();
    }

    for (size_t elementIdx = 0; elementIdx < right.size(); elementIdx++) {
      right.at(elementIdx) = generateRandomFloat();
    }

    matrix_t leftTransposed = transpose(left);
    matrix_t rightTransposed = transpose(right);

    testNonTransposed(left, right);
    testTransposeRight(left, rightTransposed);
    testTransposeLeft(leftTransposed, right);
  }
}

