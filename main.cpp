#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "common.h"
#include "cxxopts.hpp"  // Include cxxopts
#include "gguf.h"
#include "model.h"

bool verbose_g = false;
int n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);

// Helper function to replace the SentencePiece special space character with a
// regular space
std::string replace_special_space(const std::string& token) {
  std::string result = token;
  const std::string special_space = u8"\u2581";  // Unicode character ▁
  size_t pos = 0;
  while ((pos = result.find(special_space, pos)) != std::string::npos) {
    result.replace(pos, special_space.length(), " ");
    pos += 1;  // Move past the replacement
  }
  return result;
}

int main(int argc, char** argv) {
  cxxopts::Options options("llm_inference", "A simple LLM inference program.");

  options.add_options()("m,model", "Path to the GGUF model file",
                        cxxopts::value<std::string>())(
      "p,prompt", "Prompt for the model",
      cxxopts::value<std::string>()->default_value(
          "One sentence fact about silicon"))(
      "n,predict", "Number of tokens to predict",
      cxxopts::value<int>()->default_value("100"))(
      "t,threads", "Number of threads to use",
      cxxopts::value<int>()->default_value(std::to_string(n_threads)))(
      "v,verbose", "Verbose output",
      cxxopts::value<bool>()->default_value("false"))(
      "no-cnv", "Do not apply chat template",
      cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  verbose_g = result["verbose"].as<bool>();
  n_threads = result["threads"].as<int>();
  init_ops(n_threads);
  bool apply_chat_template = !result["no-cnv"].as<bool>();
  int n_predict = result["predict"].as<int>();

  std::string filename;
  if (result.count("model")) {
    filename = result["model"].as<std::string>();
  }

  std::string prompt = result["prompt"].as<std::string>();

  if (filename.empty()) {
    std::cerr << "Error: Model file not specified." << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  try {
    GGUFFile gguf_file(filename);

    if (verbose_g) {
      const auto& header = gguf_file.get_header();
      std::cout << "GGUF File Information:" << std::endl;
      std::cout << "Version: " << header.version << std::endl;
      std::cout << "Tensor count: " << header.tensor_count << std::endl;
      std::cout << "Metadata KV count: " << header.metadata_kv_count
                << std::endl;
    }

    if (verbose_g) {
      gguf_file.print_tensor_infos();
      gguf_file.print_metadata();
    }

    Model model(gguf_file);

    if (verbose_g) {
      const auto& hparams = model.hparams();
      std::cout << std::endl;
      std::cout << "Model Hyperparameters:" << std::endl;
      std::cout << "Block count: " << hparams.block_count << std::endl;
      std::cout << "Embedding length: " << hparams.embedding_length
                << std::endl;
      std::cout << "Feed forward length: " << hparams.feed_forward_length
                << std::endl;
      std::cout << "Attention head count: " << hparams.attention_head_count
                << std::endl;
      std::cout << "Attention head count KV: "
                << hparams.attention_head_count_kv << std::endl;
    }

    std::cout << std::endl;
    std::vector<int> tokens = model.tokenize(prompt, apply_chat_template);
    // Print the tokens
    if (verbose_g) {
      std::cout << "Tokenized input:" << std::endl;
      for (int token : tokens) {
        std::cout << token << " ";
      }
      std::cout << std::endl;
    }

    const auto& metadata = gguf_file.get_metadata();
    const auto& tokens_value = metadata.at("tokenizer.ggml.tokens");
    std::vector<std::string> token_strings;
    for (const auto& token : tokens_value.arr) {
      token_strings.push_back(token.str);
    }

    int end_of_turn_token_id = -1;
    for (size_t i = 0; i < token_strings.size(); ++i) {
      if (token_strings[i] == "<end_of_turn>") {
        end_of_turn_token_id = i;
        break;
      }
    }

    std::cout << "Prompt: " << prompt << "\n\n";

    std::vector<std::vector<float>> logits_vectors = model.forward(tokens, 0);
    int pos = tokens.size();
    tokens.clear();

    auto start_time = std::chrono::high_resolution_clock::now();
    int num_generated_tokens = 0;

    for (int i = 0; i < n_predict; ++i) {
      const auto& logits = logits_vectors[0];

      if (i == 0 && verbose_g) {
        std::vector<std::pair<float, int>> sorted_logits;
        for (size_t j = 0; j < logits.size(); ++j) {
          sorted_logits.push_back({logits[j], j});
        }
        std::sort(sorted_logits.begin(), sorted_logits.end());

        std::cout << std::endl;
        std::cout << "Top 10 most likely tokens:" << std::endl;
        for (int j = 0; j < 10; ++j) {
          const auto& logit_pair = sorted_logits[sorted_logits.size() - 1 - j];
          std::cout << replace_special_space(token_strings[logit_pair.second])
                    << ": " << logit_pair.first << " " << logit_pair.second
                    << std::endl;
        }
      }

      // Greedy sampling
      int next_token = std::distance(
          logits.begin(), std::max_element(logits.begin(), logits.end()));

      if (next_token == end_of_turn_token_id) {
        break;
      }

      std::cout << replace_special_space(token_strings[next_token]);
      std::cout.flush();

      num_generated_tokens++;

      if (i < n_predict - 1) {
        tokens.push_back(next_token);
        logits_vectors = model.forward(tokens, pos);
        pos++;
        tokens.clear();
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << std::endl;
    std::cout << "\nGenerated " << num_generated_tokens << " tokens in "
              << elapsed.count() << " s ("
              << num_generated_tokens / elapsed.count() << " tok/s)"
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
