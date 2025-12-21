set -e

# Only if dir doesn't exist
if [ ! -d "./third_party/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git ./third_party/llama.cpp
fi
cd ./third_party/llama.cpp
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build_dbg -j 4
cmake -B build
cmake --build build --config Release -j 4