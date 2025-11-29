set -e

bazel build -c opt //:llm_inference
if [[ "$(uname)" == "Darwin" ]]; then
    MODEL_LOC=~/Documents/temp/gemma-3-1b-it-q4_0.gguf
else
    MODEL_LOC=~/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf
fi
time bazel-bin/llm_inference --model ${MODEL_LOC} "$@"