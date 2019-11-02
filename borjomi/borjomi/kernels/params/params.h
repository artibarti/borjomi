
#pragma once

namespace borjomi {
 namespace kernels {

class conv_params;

class Params {
 public:
  Params() {}

  conv_params &conv();
};

 }
}