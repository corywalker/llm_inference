#!/bin/bash
set -e
bazel test -c opt //... --test_output=streamed 