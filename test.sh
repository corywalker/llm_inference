#!/bin/bash
set -e
bazel test -c opt //... --test_output=streamed
echo ""
echo "Unit tests passed. Now testing formatting..."
find . -path ./third_party -prune -o \( -iname '*.h' -o -iname '*.cpp' \) -print | xargs clang-format --dry-run --Werror -style=file || { echo "Code formatting issues found. Please run ./format.sh to fix."; exit 1; } 
echo "Formatting passed!"
