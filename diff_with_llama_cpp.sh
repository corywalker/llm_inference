#!/usr/bin/env bash
set -eo pipefail

if [[ "$(uname)" == "Darwin" ]]; then
    MODEL_LOC=~/Documents/temp/gemma-3-4b-it-q4_0.gguf
else
    MODEL_LOC=~/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf
fi

COMMON_ARGS=( -m "$MODEL_LOC" -p "Hello" -v -n 2 )

bazel build -c opt //:llm_inference
mkdir -p ./tmp
echo "Running llama cpp..."
# llama-eval-callback does --no-cnv by default
~/llama.cpp/bin/llama-eval-callback "${COMMON_ARGS[@]}" -s 123 > ./tmp/llama_cpp_out.txt 2>&1
sed -E -i.bak 's/^ggml_debug:[[:space:]]+//g' ./tmp/llama_cpp_out.txt
sed -i.bak 's/^                                 //g' ./tmp/llama_cpp_out.txt
rm -f ./tmp/llama_cpp_out.txt.bak
echo "Running llm_inference..."
./bazel-bin/llm_inference "${COMMON_ARGS[@]}" --no-cnv > ./tmp/llm_inference_out.txt 2>&1