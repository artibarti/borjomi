#pragma once

#include <exception>
#include <string>

namespace borjomi {

class BorjomiRuntimeException : public std::exception {
 
 private:
  std::string msg_;

 public:
  explicit BorjomiRuntimeException(const std::string &msg) : msg_(msg) {}
  
  const char *what() const throw() override {
    return msg_.c_str();
  }
};

}