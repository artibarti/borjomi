#pragma once

float sum(__m256 vec) {
  __m256 t1 = _mm256_hadd_ps(vec, vec);
  __m256 t2 = _mm256_hadd_ps(t1,t1);
  __m128 t3 = _mm256_extractf128_ps(t2,1);
  __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
  return _mm_cvtss_f32(t4);       
}

float dot(const float* left, const float* right, size_t size) {
  
  size_t nblocks  = size / 8;
  size_t nremains = size & 7;
  
  __m256 dot = _mm256_setzero_ps();
  for (size_t idx = 0; idx < nblocks; idx++) {
    const __m256 left_ = _mm256_loadu_ps(&left[idx * 8]);
    const __m256 right_ = _mm256_loadu_ps(&right[idx * 8]);
    dot = _mm256_fmadd_ps(left_, right_, dot);
  }
  
  float remains = 0;
  for (size_t remainIdx = 0; remainIdx < nremains; remainIdx++) {
    remains += left[nblocks * 8 + remainIdx] * right[nblocks * 8 + remainIdx];
  }
  return sum(dot) + remains;
}

class Vector256x {

 private:
  std::vector<__m256> data_;
  size_t length_;
  size_t xlength_;

  void load(const float* src) {
    size_t nBlocks = length_ / 8;
    size_t remains = length_ % 8;
    for (size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++) {
      data_.push_back(_mm256_loadu_ps(&src[blockIdx * 8]));
    }
    if (remains) {
      int32_t mask_src[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      };
      __m256i imask = _mm256_loadu_si256((__m256i const *)(mask_src + 8 - remains));
      data_.push_back(_mm256_maskload_ps(&src[8 * nBlocks], imask));
    }
  }

 public:
  Vector256x() {
    length_ = 0;
  }

  Vector256x(size_t length, const float* src) {
    length_ = length;
    xlength_ = (length_ / 8) + (length_ % 8 != 0 ? 1 : 0);
    load(src);
  }

  size_t length() const {
    return length_;
  }

  size_t xlength() const {
    return xlength_;
  }

  std::vector<__m256> data() {
    return data_;
  }  
};

float sumElements(__m256 vec) {
  __m256 t1 = _mm256_hadd_ps(vec, vec);
  __m256 t2 = _mm256_hadd_ps(t1,t1);
  __m128 t3 = _mm256_extractf128_ps(t2,1);
  __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
  return _mm_cvtss_f32(t4);       
}

float dot(Vector256x& left, Vector256x& right) {

  if (left.length() != right.length()) {
    throw new std::runtime_error("vec256x lenghts do not match");
  }

  __m256 dot = _mm256_setzero_ps();

  for (size_t vecIdx = 0; vecIdx < left.xlength(); vecIdx++) {
     dot = _mm256_fmadd_ps(left.data()[vecIdx], right.data()[vecIdx], dot);
  }
  return sumElements(dot);
}