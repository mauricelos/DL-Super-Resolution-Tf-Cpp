# Deep Learning Super Resolution using Tensorflow C++
### Introduction
In this project, I want to create a state of the art deep learning super resolution algorithm using only the Tensorflow C++ API.

Required Installations:

Bazel -> https://docs.bazel.build/versions/master/install-ubuntu.html (Ubuntu)

Tensorflow -> https://www.tensorflow.org/install/source

Bazel version: 0.23.1
Tensorflow version: 2.0 (git checkout r2.0)

Command to build and run dl_super_resolution:

bazel build //dl_super_resolution --define=grpc_no_ares=true
bazel run //dl_super_resolution --define=grpc_no_ares=true
