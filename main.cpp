#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
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
      "p,prompt", "Prompt for the model", cxxopts::value<std::string>())(
      "n,predict", "Number of tokens to predict",
      cxxopts::value<int>()->default_value("2048"))(
      "t,threads", "Number of threads to use",
      cxxopts::value<int>()->default_value(std::to_string(n_threads)))(
      "v,verbose", "Verbose output",
      cxxopts::value<bool>()->default_value("false"))(
      "no-cnv", "Do not apply chat template",
      cxxopts::value<bool>()->default_value("false"))(
      "i,interactive", "Interactive multi-turn chat mode",
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
  bool interactive = result["interactive"].as<bool>();
  int n_predict = result["predict"].as<int>();

  // Interactive mode is implied when no prompt is given.
  if (!result.count("prompt")) {
    interactive = true;
  }

  std::string filename;
  if (result.count("model")) {
    filename = result["model"].as<std::string>();
  }

  std::string prompt =
      result.count("prompt") ? result["prompt"].as<std::string>() : "";

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

    const auto& metadata = gguf_file.get_metadata();
    const auto& tokens_value = metadata.at("tokenizer.ggml.tokens");
    std::vector<std::string> token_strings;
    for (const auto& token : tokens_value.arr) {
      token_strings.push_back(token.str);
    }

    int end_of_turn_token_id = -1;
    for (size_t i = 0; i < token_strings.size(); ++i) {
      if (token_strings[i] == "<end_of_turn>" ||
          token_strings[i] == "<turn|>") {
        end_of_turn_token_id = i;
        break;
      }
    }

    int eos_token_id = -1;
    if (metadata.count("tokenizer.ggml.eos_token_id")) {
      eos_token_id = (int)metadata.at("tokenizer.ggml.eos_token_id").scalar.u32;
    }

    int think_token_id = -1;
    int channel_token_id = -1;
    for (size_t i = 0; i < token_strings.size(); ++i) {
      const std::string& ts = token_strings[i];
      if (ts == "<|channel>thought") {
        think_token_id = i;
      } else if ((ts == "<|think|>" || ts == "<think>") &&
                 think_token_id == -1) {
        think_token_id = i;
      }
      if (ts == "<channel|>") {
        channel_token_id = i;
      } else if ((ts == "<|channel|>" || ts == "</think>") &&
                 channel_token_id == -1) {
        channel_token_id = i;
      }
    }

    // Run generation from a pre-tokenized prompt. Returns the response text
    // (thinking content is excluded from the returned string).
    auto generate = [&](const std::vector<int>& prompt_tokens,
                        bool starts_thinking) -> std::string {
      bool is_thinking = starts_thinking;
      if (is_thinking) {
        std::cout << "\x1b[90m[Start thinking]\n";
      }

      if (verbose_g) {
        if (starts_thinking) {
          // Top-10 logits are less useful for reasoning prefill; skip.
        }
        std::cout << "Prompt tokens: " << prompt_tokens.size() << std::endl;
      }

      model.reset_kv_cache();
      std::vector<int> tokens = prompt_tokens;
      auto logits_vectors = model.forward(tokens, 0);
      int pos = (int)tokens.size();
      tokens.clear();

      auto start_time = std::chrono::high_resolution_clock::now();
      int num_generated = 0;
      std::string response;
      bool in_response = !starts_thinking;

      for (int i = 0; i < n_predict; ++i) {
        const auto& logits = logits_vectors[0];

        if (i == 0 && verbose_g && !starts_thinking) {
          std::vector<std::pair<float, int>> sorted_logits;
          for (size_t j = 0; j < logits.size(); ++j) {
            sorted_logits.push_back({logits[j], (int)j});
          }
          std::sort(sorted_logits.begin(), sorted_logits.end());
          std::cout << "\nTop 10 most likely tokens:" << std::endl;
          for (int j = 0; j < 10; ++j) {
            const auto& p = sorted_logits[sorted_logits.size() - 1 - j];
            std::cout << replace_special_space(token_strings[p.second]) << ": "
                      << p.first << " " << p.second << std::endl;
          }
        }

        int next_token = std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));

        if (next_token == end_of_turn_token_id || next_token == eos_token_id) {
          break;
        }

        if (verbose_g) {
          std::cout << "\nGenerated Token ID: " << next_token << " String: \""
                    << token_strings[next_token] << "\"" << std::endl;
        }

        if (next_token == think_token_id) {
          is_thinking = true;
          std::cout << "\x1b[90m\n[Start thinking]\n";
        } else if (next_token == channel_token_id) {
          is_thinking = false;
          in_response = true;
          std::cout << "\x1b[0m\n[End thinking]\n\n";
        } else {
          std::string tok = replace_special_space(token_strings[next_token]);
          std::cout << tok;
          if (in_response) response += tok;
        }
        std::cout.flush();
        num_generated++;

        if (i < n_predict - 1) {
          tokens.push_back(next_token);
          logits_vectors = model.forward(tokens, pos);
          pos++;
          tokens.clear();
        }
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      if (is_thinking) std::cout << "\x1b[0m";
      std::cout << "\n\nGenerated " << num_generated << " tokens in "
                << elapsed.count() << " s (" << num_generated / elapsed.count()
                << " tok/s)" << std::endl;
      return response;
    };

    if (interactive) {
      std::vector<ChatMessage> history;
      // Seed history with -p if provided.
      if (!prompt.empty()) {
        history.push_back({"user", prompt});
      }

      std::cout << "Interactive chat (Ctrl-D to quit)\n" << std::endl;
      while (true) {
        if (history.empty() || history.back().role != "user") {
          std::cout << "\x1b[1m> \x1b[0m";
          std::cout.flush();
          std::string user_input;
          if (!std::getline(std::cin, user_input)) break;
          if (user_input.empty()) continue;
          history.push_back({"user", user_input});
        }

        std::cout << std::endl;
        bool prefilled_thinking = false;
        std::vector<int> tokens = model.tokenize_chat(
            history, /*enable_thinking=*/true, &prefilled_thinking);
        if (verbose_g) {
          std::cout << "Tokenized conversation: " << tokens.size() << " tokens"
                    << std::endl;
        }

        std::string response = generate(tokens, prefilled_thinking);
        history.push_back({"assistant", response});
        std::cout << std::endl;
      }
    } else {
      // Single-turn mode.
      std::cout << "Prompt: " << prompt << "\n\n";
      bool prefilled_thinking = false;
      std::vector<int> tokens;
      if (apply_chat_template) {
        tokens =
            model.tokenize_chat({{"user", prompt}},
                                /*enable_thinking=*/true, &prefilled_thinking);
      } else {
        tokens = model.tokenize(prompt, false, &prefilled_thinking);
      }
      if (verbose_g) {
        std::cout << "Tokenized input:" << std::endl;
        for (int t : tokens) std::cout << t << " ";
        std::cout << std::endl;
      }
      generate(tokens, prefilled_thinking);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
