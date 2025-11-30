config_setting(
    name = "macos_arm64",
    values = {"cpu": "darwin_arm64"},
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
      "//...": "",
    },
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    visibility = ["//:__subpackages__"],
)

cc_library(
    name = "gguf",
    srcs = ["gguf.cpp"],
    hdrs = ["gguf.h"],
    visibility = ["//:__subpackages__"],
    deps = [":common"],
)

cc_library(
    name = "ops",
    srcs = ["ops.cpp"],
    hdrs = ["ops.h", "thread_pool.h"],
    copts = select({
        "//:macos_arm64": [],
        "//conditions:default": ["-mavx2", "-mfma"],
    }),
    visibility = ["//:__subpackages__"],
    deps = [":common", ":gguf"],
)

cc_library(
    name = "model",
    srcs = ["model.cpp"],
    hdrs = ["model.h", "tensor.h"],
    deps = [
        ":gguf",
        ":ops",
        ":common",
    ],
    visibility = ["//:__subpackages__"],
)

cc_binary(
    name = "llm_inference",
    srcs = ["main.cpp"],
    deps = [
        ":model",
        "//third_party/cxxopts:cxxopts",
        ":common",
    ],
)

cc_test(
    name = "gguf_test",
    size = "small",
    srcs = ["gguf_test.cpp"],
    deps = [
        ":gguf",
        ":ops",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.cpp"],
    deps = [
        ":ops",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cpp"],
    deps = [
        ":model",
        "@googletest//:gtest_main",
    ],
)