# Deep Learning Super Resolution using Tensorflow C++
### Introduction
In this project, I want to create a state of the art deep learning super resolution algorithm using only the Tensorflow C++ API.

#### Required Installations
- Bazel -> [Bazel Install Guide](https://docs.bazel.build/versions/master/install.html)
- Tensorflow -> [Tensorflow Install Guide from source](https://www.tensorflow.org/install/source)

#### Used Setup
- Bazel: v.0.23.0
- Tensorflow: v.2.0 (git checkout r2.0)
- Googletest: v.1.8.x (git checkout v.1.8.x)
- CUDA: v.9.0
- Cudnn: v.7.5
- gcc/g++: v.4.8

#### Steps required before first build
- All dependencies are installed
- Run ./configure inside tensorflow repo
- Concatenate .bazelrc to .tf_configure.bazelrc to fix bug in Tensorflow v.2.0
    - ```cat tensorflow/.bazelrc >> tensorflow/.tf_configure.bazelrc ```


#### Command to build, run and test the project
```Python
bazel build //dl_super_resolution:dl_super_resolution
bazel run //dl_super_resolution:dl_super_resolution
bazel test //dl_super_resolution:dl_super_resolution_tests
```

#### IDE Setup
- IDE: VSCode
- Extensions:
    - [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
    - [vscode-bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)
    - [C++ Intellisense](https://marketplace.visualstudio.com/items?itemName=austin.code-gnu-global)
    - [Markdown Preview Github Styling](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-preview-github-styles)

#### TF Configure bazelrc (for reference)

```C++
build --action_env PYTHON_BIN_PATH="/path/to/bin/python"
build --action_env PYTHON_LIB_PATH="/path/to/lib/python2.7/site-packages"
build --python_path="/path/to/bin/python"
build:xla --define with_xla_support=true
build --action_env TF_NEED_OPENCL_SYCL="0"
build --action_env TF_NEED_ROCM="0"
build --action_env TF_NEED_CUDA="1"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-9.0"
build --action_env TF_CUDA_VERSION="9.0"
build --action_env CUDNN_INSTALL_PATH="/usr/local/cuda-9.0"
build --action_env TF_CUDNN_VERSION="7"
build --action_env TF_NCCL_VERSION=""
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="6.1"
build --action_env LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
build --action_env TF_CUDA_CLANG="0"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/gcc-4.8"
build --config=cuda
test --config=cuda
build:opt --copt=-march=native
build:opt --copt=-Wno-sign-compare
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
build:v2 --define=tf_api_version=2
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test --test_tag_filters=-benchmark-test,-no_oss,-oss_serial
test --build_tag_filters=-benchmark-test,-no_oss
test --test_tag_filters=-gpu
test --build_tag_filters=-gpu
```