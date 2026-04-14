load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "inja",
    hdrs = ["inja/inja.hpp"],
    strip_include_prefix = "/third_party/inja",
    visibility = ["//visibility:public"],
    deps = ["//third_party/nlohmann_json:nlohmann_json"],
)
