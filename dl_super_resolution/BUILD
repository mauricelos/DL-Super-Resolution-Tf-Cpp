load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "dl_super_resolution",
    srcs = [
        "dummy.cpp",
    ],
    copts = ["-Iexternal/org_tensorflow"],
    linkopts = [
        "-lm",
        "-Wl,--no-as-needed",
        "-Wl,--allow-multiple-definition",
        "-Wl,--whole-archive",
    ],
    deps = [
        "@org_tensorflow//tensorflow:libtensorflow_cc.so",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)
