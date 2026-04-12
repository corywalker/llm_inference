set -e

bazel build -c opt //:llm_inference
if [[ "$(uname)" == "Darwin" ]]; then
    MODEL_LOC="$HOME/.cache/lm-studio/models/lmstudio-community/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf"
else
    MODEL_LOC=~/gemma-3-4b-it-Q4_K_M.gguf
fi
time bazel-bin/llm_inference --model ${MODEL_LOC} "$@"