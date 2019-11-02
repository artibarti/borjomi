/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <stdlib.h>
#include <string>
#include <utility>

#ifdef _WIN32
  #include <malloc.h>
#endif

#ifdef __MINGW32__
  #include <mm_malloc.h>
#endif

namespace borjomi {

template<typename T, size_t alignment>
class AlignedAllocator {

 public:
  typedef T value_type;
  typedef T *pointer;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T &reference;
  typedef const T &const_reference;
  typedef const T *const_pointer;

  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U, alignment> other;
  };

  AlignedAllocator() {}

  template<typename U>
  AlignedAllocator(const AlignedAllocator<U, alignment>&) {}

  const_pointer address(const_reference value) const {
    return std::addressof(value);
  }

  pointer address(reference value) const {
    return std::addressof(value);
  }

  pointer allocate(size_type size, const void* = nullptr) {
    void *p = alignedAlloc(alignment, sizeof(T) * size);
    if (!p && size > 0) {
      throw std::runtime_error("Failed to allocate memory");
    }
    return static_cast<pointer>(p);
  }

  size_type max_size() const {
    return ~static_cast<std::size_t>(0) / sizeof(T);
  }

  void deallocate(pointer ptr, size_type) {
    aligned_free(ptr);
  }

  template <class U, class V>
  void construct(U *ptr, const V &value) {
    void *p = ptr;
    ::new (p) U(value);
  }

  template <class U, class... Args>
  void construct(U *ptr, Args &&... args) {
    void *p = ptr;
    ::new (p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void construct(U *ptr) {
    void *p = ptr;
    ::new (p) U();
  }

  template <class U>
  void destroy(U *ptr) {
    ptr->~U();
  }

 private:
  void *alignedAlloc(size_type align, size_type size) const {
    
    #if defined(_MSC_VER)
        return ::_aligned_malloc(size, align);
    #elif defined(__ANDROID__)
        return ::memalign(align, size);
    #elif defined(__MINGW32__)
        return ::_mm_malloc(size, align);
    #else
      void *p;
      if (::posix_memalign(&p, align, size) != 0) {
        p = 0;
      }
      return p;
    #endif
  }

  void aligned_free(pointer ptr) {
    #if defined(_MSC_VER)
      ::_aligned_free(ptr);
    #elif defined(__MINGW32__)
      ::_mm_free(ptr);
    #else
      ::free(ptr);
    #endif
  }
};

template<typename T, typename U, std::size_t alignment>
bool operator==(const AlignedAllocator<T, alignment>&, const AlignedAllocator<U, alignment>&) {
  return true;
}

template<typename T, typename U, std::size_t alignment>
bool operator!=(const AlignedAllocator<T, alignment>&, const AlignedAllocator<U, alignment>&) {
  return false;
}

}