
## project borjomi ##

-----------------

**project borjomi** is a lightweight neural network framework, implemented in a well fragmented dependency-free way. The project was written in C++ and it is still under active development. I welcome any suggestions and critics.

The project based on another deep learning framework, called Tiny DNN. Due to that, in many aspects the project show similarity with that one, and in some cases may use code snippets from that one. In these cases the the original license is provided at the top of the source files.

Take a look for Tiny DNN [here](https://github.com/tiny-dnn/tiny-dnn).

## summary

Project Borjomi is a project providing an implementation of deep learning in alightweight, dependency-free way. The project was written in C++ and it is still under active development. I welcome suggestions, critics, and anyone may feel free to contact me on the channels available on the github page of the project. 

My main goal was to provide a framework which allows users to play around with the 
basic functionality of neural networks, and also to let developers implement their own versions of the different layers easily by implementing the provided interfaces and that way make the framework ‘hackable’. To be efficient about this aim, the project was designed in a clean, easily-manageable and well-fragmented way, so the different parts of the framework can be reimagined and reused with only a little effort. This is still one of the most important advantage of Project Borjomi.

## usage

Borjomi is a header-only, and if you do not need any additional speed up options, it can be used simply without any configutration. All you need is a C++ compiler with c++14 support. You can import the whole library as a single-header dependency by including the header file "borjomi/borjomi.h", which contains everything you need. Sure, you should tell your compiler the include path for borjomi. Should look something like this:

  `g++ -I/home/myHome/borjomi/ myProgram.cpp -o myProgram`

**CMake support**

You can use cmake to make your life easier. Especiall when you want to use borjomi on different platforms, let's say gpu, or you want to increase performance by using AVX / AVX2, you may need to pass additional flags to your compiler. To make it easy, you can configure your compilation with cmake. If you want to use AVX, just tell cmake, and it will handle the rest.

I suggest you to create a build folder for the cmake result. It can be right next to the borjomi folder (the one contains the `cmake`, `example` folders). Entering it and exeture cmake like that

  `cmake .. -DUSE_AVX=true`

will allow AVX support. With other parallelization options it is so similar.ou can take a look at the CMakeLists.txt to see what is happening under the hood. Also, you can configure default values there.

## features

Borjomi is suitable for different kinds of neural networks with some constraints. It must be a sequential one, graph networks are not supported yet. Since borjomi is still under development, the number of supported layers are also limited, but can be enough already for complex network constructions. Besides them, you can puzzle up almost any kind of configuration. To see how a network construction looks like, check the `example` folder for some code.

**supported layers**

 - Trainables
    * Fully Connected layer
    * Convolutional Layer

- Pooling Layers
    * Max Pooling Layer
    * Min Pooling Layer
    * Average Pooling Layer

- Activations
    * tanh
    * asinh
    * sigmoid
    * softmax
    * softplus
    * softsign
    * rectified linear (relu)
    * leaky relu
    * identity
    * scaled tanh
    * exponential linear units (elu)
    * scaled exponential linear units (selu) 
 
**parallelisation support**

The framework can be accelerated with massive parallelism. It supports various architectures. It is importantto note here that some of them is just partly supported or still under the integration process.

Supported architectures:

- AVX / AVX2
    * Fully Connected Layer
    * Convolutional Layer
- Native C++ multithreading
    * Fully Connected Layer
    * Convolutional Layer
- CUDA
    * Fully Connected Layer

Under integration (not ready to use):

- Intel Xeon Phi
- AVX 512
- Intel MKL