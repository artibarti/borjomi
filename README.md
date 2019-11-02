
## project borjomi ##

-----------------

**project borjomi** is a lightweight neural network framework, implemented in a well fragmented dependency-free way. The project was written in C++ and it is still under active development. I welcome any suggestions and critics.

The project based on another deep learning framework, called Tiny DNN. Due to that, in many aspects the project show similarity with that one, and in some cases may use code snippets from that one. In these cases the the original license is provided at the top of the source files.

Take a look for Tiny DNN [here](https://github.com/tiny-dnn/tiny-dnn).

**summary**

Project Borjomi is a project providing an implementation of deep learning in alightweight, dependency-free way. The project was written in C++ and it is still under active development. I welcome suggestions, critics, and anyone may feel free to contact me on the channels available on the github page of the project. 

My main goal was to provide a framework which allows users to play around with the 
basic functionality of neural networks, and also to let developers implement their own versions of the different layers easily by implementing the provided interfaces and that way make the framework ‘hackable’. To be efficient about this aim, the project was designed in a clean, easily-manageable and well-fragmented way, so the different parts of the framework can be reimagined and reused with only a little effort. This is still one of the most important advantage of Project Borjomi.

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

The framework can be accelerated with massive parallelism. It supports variousarchitectures.