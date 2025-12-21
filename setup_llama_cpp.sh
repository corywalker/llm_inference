set -e

# Only if dir doesn't exist
if [ ! -d "./third_party/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ./third_party/llama.cpp
fi
cd ./third_party/llama.cpp

# Turn off Metal because we don't support this yet in llm_inference.
# Turn off llamafile because it causes a different llamafile_sgemm kernel to be used for prefill:
# https://github.com/ggml-org/llama.cpp/blob/9496bbb8081463b16412defcb4ef1ed17f7fb42f/ggml/src/ggml-cpu/llamafile/sgemm.cpp#L2595-L2612
# For simplicity, I just want the standard kernels used across prefill and decode. So, we disable it.
COMMON_CMAKE_FLAGS="-DGGML_METAL=OFF -DGGML_LLAMAFILE=OFF"
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug $COMMON_CMAKE_FLAGS
cmake --build build_dbg -j 4
cmake -B build $COMMON_CMAKE_FLAGS
cmake --build build --config Release -j 4