load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "dl_super_resolution",
    srcs = [
        "include/dl_model_helper.h",
        "src/dl_model_helper.cpp",
        "src/dummy.cpp",  #
    ],
    deps = [
        "@org_tensorflow//tensorflow:libtensorflow_cc.so",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "dl_super_resolution_tests",
    srcs = glob(["test/*.cpp"]),
    deps = [
        "dl_super_resolution",
        "@googletest//:gtest_main",
    ],
)
