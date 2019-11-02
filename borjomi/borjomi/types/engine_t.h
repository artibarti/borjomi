#pragma once

#include <iostream>
#include <string>

namespace borjomi {

enum class engine_t {
  internal, nnpack, libdnn, avx, opencl, cblas,
  intel_mkl, sse, tbb, threads, cuda, phi
};

std::string toString(engine_t engine) {
  switch (engine) {
		case engine_t::internal:  return "Internal";
    case engine_t::nnpack:    return "NNPack";
    case engine_t::avx:       return "AVX";
    case engine_t::opencl:    return "OpenCL";
    case engine_t::cblas:     return "CBLAS";
    case engine_t::intel_mkl: return "Intel MKL";     
    case engine_t::libdnn:    return "LibDNN";
    case engine_t::sse:       return "SSE";
    case engine_t::tbb:       return "TBB";
    case engine_t::threads:   return "threads";
    case engine_t::cuda:      return "cuda";
    case engine_t::phi:      return "Intel Xeon Phi";
		default: return "Unknown";
  }
}

std::ostream& operator<<(std::ostream& os, engine_t engine) {
  os << toString(engine);
  return os;
}

}