#!/usr/bin/env bash
set -eo pipefail

# LLAMA_EVAL_CALLBACK_BIN is https://github.com/ggml-org/llama.cpp/blob/master/examples/eval-callback/README.md
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <MODEL_LOCATION> <LLAMA_EVAL_CALLBACK_BINARY_PATH>"
    exit 1
fi

MODEL_LOC="$1"
LLAMA_EVAL_CALLBACK_BIN="$2"

COMMON_ARGS=( -m "$MODEL_LOC" -p "Hello" -v -n 2 )

bazel build -c opt //:llm_inference
mkdir -p ./tmp
echo "Running llama cpp..."
# llama-eval-callback does --no-cnv by default
"$LLAMA_EVAL_CALLBACK_BIN" "${COMMON_ARGS[@]}" -s 123 > ./tmp/llama_cpp_out.txt 2>&1
sed -E -i.bak 's/^ggml_debug:[[:space:]]+//g' ./tmp/llama_cpp_out.txt
sed -i.bak 's/^                                 //g' ./tmp/llama_cpp_out.txt
rm -f ./tmp/llama_cpp_out.txt.bak
echo "Running llm_inference..."
./bazel-bin/llm_inference "${COMMON_ARGS[@]}" --no-cnv > ./tmp/llm_inference_out.txt 2>&1