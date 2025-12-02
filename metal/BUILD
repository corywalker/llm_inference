package(default_visibility = ["//visibility:public"])

cc_library(
    name = "metal_ops",
    srcs = [
        "context.cpp",
        "ops.cpp",
    ],
    hdrs = [
        "context.h",
        "ops.h",
    ],
    data = [
        "kernels.metal",
    ],
    copts = [
        "-x", "objective-c++",
        "-fobjc-arc",
        "-std=c++17",
    ],
    linkopts = [
        "-framework Metal",
        "-framework Foundation",
    ],
    deps = [
        "//:common",
        "//:gguf",
    ],
)
