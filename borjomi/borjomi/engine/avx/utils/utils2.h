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