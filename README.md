# Deep Learning Super Resolution using Tensorflow C++
### Introduction
In this project, I want to create a state of the art deep learning super resolution algorithm using only the Tensorflow C++ API.

#### Required Installations
- Bazel -> [Bazel Install Guide](https://docs.bazel.build/versions/master/install.html)
- Tensorflow -> [Tensorflow Install Guide from source](https://www.tensorflow.org/install/source)

#### Used Setup
- Bazel: v.0.23.0
- Tensorflow: v.2.0 (git checkout r2.0)
- CUDA: v.9.0
- Cudnn: v.7.5
- gcc/g++: v.4.8

#### Command to build and run the project:
- bazel build //dl_super_resolution:dl_super_resolution
- bazel run //dl_super_resolution:dl_super_resolution

#### IDE Setup
- IDE: VSCode
- Extensions:
    - [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
    - [vscode-bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)
    - [C++ Intellisense](https://marketplace.visualstudio.com/items?itemName=austin.code-gnu-global)
    - [Markdown Preview Github Styling](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-preview-github-styles)
    
