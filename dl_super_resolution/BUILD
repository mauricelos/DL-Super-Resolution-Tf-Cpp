load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "dl_super_resolution_lib",
    srcs = [
        "src/dl_model_helper.cpp",
        "src/dummy.cpp",
    ],
    hdrs = ["include/dl_model_helper.h"],
    data = glob(["images/**"]),
    includes = [
        "include",
        "src",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        "@opencv",
        "@org_tensorflow//tensorflow:libtensorflow_cc.so",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "dl_super_resolution_tests",
    srcs = glob(["test/*.cpp"]),
    deps = [
        ":dl_super_resolution_lib",
        "@googletest//:gtest_main",
    ],
)
