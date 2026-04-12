set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: ./run.sh <model_path> [extra_args...]"
    exit 1
fi
MODEL_LOC=$1
shift

bazel build -c opt //:llm_inference
time bazel-bin/llm_inference --model "${MODEL_LOC}" "$@"