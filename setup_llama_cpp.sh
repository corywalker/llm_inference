set -e

# Only if dir doesn't exist
if [ ! -d "./third_party/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ./third_party/llama.cpp
fi
cd ./third_party/llama.cpp

COMMON_CMAKE_FLAGS="-DGGML_METAL=OFF -DGGML_BLAS=OFF"
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug $COMMON_CMAKE_FLAGS
cmake --build build_dbg -j 4
cmake -B build $COMMON_CMAKE_FLAGS
cmake --build build --config Release -j 4