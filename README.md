# C++ Inference for Large Language Models

This project is a C++ implementation for running large language model inference from GGUF, inspired by `llama.cpp`. It provides a lightweight and efficient way to run inference. The project focuses on simplicity over performance, and is not intended for production use.

## Features

- **KV Cache** for efficient decoding.
- **Vectorized matrix multiplication** using AVX2 (x86) and NEON (ARM) instructions.
- **Q4_0 quantization only**.
- **CPU only** - optimized for x86 and Apple Silicon.
- **Simple architecture** - no computation graph, sequential execution.
- **Greedy sampling**.

## Tested Models

This project has been tested with:
- [google/gemma-3-1b-it-qat-q4_0-gguf/blob/main/gemma-3-1b-it-q4_0.gguf](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/blob/main/gemma-3-1b-it-q4_0.gguf)
- [google/gemma-3-4b-it-qat-q4_0-gguf/blob/main/gemma-3-4b-it-q4_0.gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/blob/main/gemma-3-4b-it-q4_0.gguf)

Known not working:
- [unsloth/gemma-3-270m-it-GGUF/blob/main/gemma-3-270m-it-Q4_0.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/blob/main/gemma-3-270m-it-Q4_0.gguf)

## Getting Started

### Prerequisites

- A C++ compiler that supports C++17 (GCC, Clang, MSVC).
- `bazel` build tool.

### Building

The project uses Bazel for building. To build the optimized binary:

```bash
bazel build -c opt //:llm_inference
```

The binary will be located at `bazel-bin/llm_inference`.

### Running Inference

To run inference, provide the path to your GGUF model:

```bash
./bazel-bin/llm_inference --model /path/to/your/model.gguf [options]
```

#### Options
- `-m, --model`: Path to the GGUF model file (required).
- `-p, --prompt`: Prompt for the model (default: "One sentence fact about silicon").
- `-n, --predict`: Number of tokens to predict (default: 100).
- `-t, --threads`: Number of threads to use (default: half of available cores).
- `-v, --verbose`: Enable verbose output (default: false).
- `--no-cnv`: Do not apply chat template (default: false).
- `-h, --help`: Print usage information.

**Example:**

```bash
./bazel-bin/llm_inference --model gemma-3-1b-it-q4_0.gguf --prompt "Explain quantum computing"
```

```
Prompt: Explain quantum computing

Okay, let's break down quantum computing – it's a fascinating and incredibly complex topic, and it's essentially a new way to think about how computers can be different from the traditional ones we're used today. Here's the core idea:

**What is it?**

**1.  It's not just about bits and bytes, it's about quantum mechanics.**

Traditional computers use bits, which are like a single, 0 or 1,

Generated 100 tokens in 10.9122 s (9.16401 tok/s)
```

### Running Tests

To run the test suite:

```bash
./test.sh
```

### Codespace Setup

If you are using a GitHub Codespace, run the setup script to install dependencies:

```bash
./setup.sh
```

## License

This project is open source. Portions of the code (specifically GGUF parsing and some operations) are adapted from `llama.cpp` (MIT License).
