#pragma once

#include "borjomi/types/types.h"
#include "borjomi/util/util.h"
#include "borjomi/engine/internal/kernels/kernels.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "borjomi/engine/cuda/helper_cuda.h"

namespace borjomi {
 namespace engine {
  namespace cuda {

  void multiply(float alpha, bool transLeft, size_t rowsInLeft, size_t colsInLeft, const float* left,
    bool transRight, size_t rowsInRight, size_t colsInRight, const float* right, float beta, float* result) {

    #ifdef USE_OPENCL

      cublasHandle_t handler;
      checkCudaErrors(cublasCreate(&handler));

      float *dLeft, *dRight, *dResult;
      
      size_t rowsInRes, colsInRes;
      rowsInRes = transLeft ? colsInLeft : rowsInLeft;
      colsInRes = transRight ? rowsInRight : colsInRight;

      checkCudaErrors(cudaMalloc((void **) &dLeft, sizeof(float) * rowsInLeft * colsInLeft));
      checkCudaErrors(cudaMalloc((void **) &dRight, sizeof(float) * rowsInRight * colsInRight));
      checkCudaErrors(cudaMalloc((void **) &dResult, sizeof(float) * rowsInRes * colsInRes));

      checkCudaErrors(cudaMemcpy(dLeft, left, sizeof(float) * rowsInLeft * colsInLeft, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dRight, right, sizeof(float) * rowsInRight * colsInRight, cudaMemcpyHostToDevice));
      if (beta) {
        checkCudaErrors(cudaMemcpy(dResult, result, sizeof(float) * rowsInRes * colsInRes, cudaMemcpyHostToDevice));
      }

      if (transLeft) {
        checkCudaErrors(cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_T, colsInRight, colsInLeft,
          rowsInLeft, &alpha, dRight, colsInRight, dLeft, colsInLeft, &beta, dResult, colsInRes));
      } else if (transRight) {
        checkCudaErrors(cublasSgemm(handler, CUBLAS_OP_T, CUBLAS_OP_N, rowsInRight, rowsInLeft,
          colsInLeft, &alpha, dRight, colsInRight, dLeft, colsInLeft, &beta, dResult, colsInRes));
      } else {
        checkCudaErrors(cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, colsInRight, rowsInLeft,
          colsInLeft, &alpha, dRight, colsInRight, dLeft, colsInLeft, &beta, dResult, colsInRes));
      }
      checkCudaErrors(cudaMemcpy(result, dResult, sizeof(float) * rowsInRes * colsInRes, cudaMemcpyDeviceToHost));

      checkCudaErrors(cublasDestroy(handler));
      checkCudaErrors(cudaFree(dLeft));
      checkCudaErrors(cudaFree(dRight));
      checkCudaErrors(cudaFree(dResult));

    #else
      throw BorjomiRuntimeException("Cuda support is not available");
    #endif
  }
  
  }
 }
}