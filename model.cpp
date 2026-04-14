#include "model.h"

#include <cmath>
#include <inja/inja.hpp>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "tensor.h"

Model::Model(GGUFFile& gguf_file) : gguf_file_(gguf_file) {
  token_embd_weight_ = nullptr;
  output_norm_weight_ = nullptr;
  token_embd_per_layer_weight_ = nullptr;
  per_layer_model_proj_weight_ = nullptr;
  per_layer_proj_norm_weight_ = nullptr;

  load_hparams(gguf_file);
  layers_.resize(hparams_.block_count);
  for (auto& layer : layers_) {
    layer.attn_norm_weight = nullptr;
    layer.attn_q_weight = nullptr;
    layer.attn_k_weight = nullptr;
    layer.attn_v_weight = nullptr;
    layer.attn_output_weight = nullptr;
    layer.ffn_norm_weight = nullptr;
    layer.ffn_gate_weight = nullptr;
    layer.ffn_up_weight = nullptr;
    layer.ffn_down_weight = nullptr;
    layer.post_attention_norm_weight = nullptr;
    layer.post_ffw_norm_weight = nullptr;
    layer.attn_k_norm_weight = nullptr;
    layer.attn_q_norm_weight = nullptr;
    layer.out_scale_weight = nullptr;
    layer.per_layer_inp_gate_weight = nullptr;
    layer.per_layer_proj_weight = nullptr;
    layer.per_layer_post_norm_weight = nullptr;
  }

  kv_cache_.resize(hparams_.block_count);
  map_tensors(gguf_file);
  load_vocabulary();

  if (token_embd_weight_->tensor_type == (uint32_t)GGUFTensorType::F16) {
    LOG_VERBOSE("Loading token embedding weights as F16...");
    const size_t num_elements =
        token_embd_weight_->shape[0] * token_embd_weight_->shape[1];
    token_embd_weight_f16_.resize(num_elements);
    gguf_file_.read_tensor_data(*token_embd_weight_,
                                token_embd_weight_f16_.data(),
                                num_elements * sizeof(uint16_t));
    LOG_VERBOSE("Loading done.");
  }
}

void Model::load_hparams(GGUFFile& gguf_file) {
  const auto& metadata = gguf_file.get_metadata();
  auto get_key = [&](const std::string& key,
                     bool die_if_not_found = true) -> const GGUFValue* {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
      if (die_if_not_found) {
        std::cerr << "Failed to find metadata key: " << key << std::endl;
        exit(1);
      }
      return nullptr;
    }
    return &it->second;
  };

  hparams_.architecture = get_key("general.architecture")->str;
  std::string arch = hparams_.architecture;

  hparams_.block_count = get_key(arch + ".block_count")->scalar.u32;
  hparams_.embedding_length = get_key(arch + ".embedding_length")->scalar.u32;
  hparams_.feed_forward_length =
      get_key(arch + ".feed_forward_length")->scalar.u32;
  hparams_.attention_head_count =
      get_key(arch + ".attention.head_count")->scalar.u32;
  hparams_.attention_head_count_kv =
      get_key(arch + ".attention.head_count_kv")->scalar.u32;
  hparams_.f_norm_rms_eps =
      get_key(arch + ".attention.layer_norm_rms_epsilon")->scalar.f32;
  hparams_.rope_freq_base = get_key(arch + ".rope.freq_base")->scalar.f32;
  // Hardcoding this to 1, since it seems possible that that llama.cpp ignores
  // the rope.scaling.factor of 8 on the larger Gemma model because the
  // rope_freq_base is already set to 1,000,000, which inherently handles the
  // extended context. Forcing the scaling factor to 1.0 which is a bit of a
  // hack, but it certainly helps with getting more accurate results.
  hparams_.rope_freq_scale = 1.0f;
  hparams_.n_embd_head_k =
      hparams_.embedding_length / hparams_.attention_head_count;
  const auto* attention_key_length_value =
      get_key(arch + ".attention.key_length", false);
  if (attention_key_length_value) {
    hparams_.n_embd_head_k = attention_key_length_value->scalar.u32;
  }

  const auto* attention_key_length_swa_value =
      get_key(arch + ".attention.key_length_swa", false);
  hparams_.n_embd_head_k_swa = attention_key_length_swa_value
                                   ? attention_key_length_swa_value->scalar.u32
                                   : hparams_.n_embd_head_k;

  const auto* attention_value_length_value =
      get_key(arch + ".attention.value_length", false);
  hparams_.n_embd_head_v = attention_value_length_value
                               ? attention_value_length_value->scalar.u32
                               : hparams_.n_embd_head_k;

  const auto* attention_value_length_swa_value =
      get_key(arch + ".attention.value_length_swa", false);
  hparams_.n_embd_head_v_swa =
      attention_value_length_swa_value
          ? attention_value_length_swa_value->scalar.u32
          : hparams_.n_embd_head_v;

  hparams_.f_attention_scale = 1.0f / std::sqrt(float(hparams_.n_embd_head_k));
  if (arch == "gemma4") {
    hparams_.f_attention_scale = 1.0f;
  }

  const auto* max_alibi_bias_value =
      get_key(arch + ".attention.max_alibi_bias", false);
  hparams_.f_max_alibi_bias =
      max_alibi_bias_value ? max_alibi_bias_value->scalar.f32 : 0.0f;

  const auto* attn_soft_cap_value =
      get_key(arch + ".attention.logit_softcapping", false);
  hparams_.attn_soft_cap =
      attn_soft_cap_value ? attn_soft_cap_value->scalar.f32 : 0.0f;

  const auto* swa_layers_value =
      get_key(arch + ".attention.sliding_window_pattern", false);
  if (swa_layers_value && swa_layers_value->type == GGUFType::ARRAY) {
    for (const auto& val : swa_layers_value->arr) {
      hparams_.swa_layers.push_back(val.scalar.b);
    }
  }

  const auto* final_logit_softcap_value =
      get_key(arch + ".attention.final_logit_softcapping", false);
  hparams_.final_logit_softcap =
      final_logit_softcap_value ? final_logit_softcap_value->scalar.f32 : 0.0f;

  const auto* embedding_length_per_layer_value =
      get_key(arch + ".embedding_length_per_layer", false);
  if (!embedding_length_per_layer_value) {
    embedding_length_per_layer_value =
        get_key(arch + ".embedding_length_per_layer_input", false);
  }
  hparams_.embedding_length_per_layer =
      embedding_length_per_layer_value
          ? embedding_length_per_layer_value->scalar.u32
          : 0;

  const auto* shared_kv_layers_value =
      get_key(arch + ".attention.shared_kv_layers", false);
  if (shared_kv_layers_value) {
    hparams_.n_layer_kv_from_start =
        hparams_.block_count - shared_kv_layers_value->scalar.u32;
  } else {
    hparams_.n_layer_kv_from_start = -1;
  }
}

void Model::map_tensors(GGUFFile& gguf_file) {
  auto& tensor_infos =
      const_cast<std::vector<TensorInfo>&>(gguf_file.get_tensor_infos());

  for (auto& tensor : tensor_infos) {
    if (tensor.name == "token_embd.weight") {
      token_embd_weight_ = &tensor;
    } else if (tensor.name == "output_norm.weight") {
      output_norm_weight_ = &tensor;
    } else if (tensor.name == "token_embd_per_layer.weight" ||
               tensor.name == "per_layer_token_embd.weight") {
      token_embd_per_layer_weight_ = &tensor;
    } else if (tensor.name == "per_layer_model_proj.weight") {
      per_layer_model_proj_weight_ = &tensor;
    } else if (tensor.name == "per_layer_proj_norm.weight") {
      per_layer_proj_norm_weight_ = &tensor;
    } else if (tensor.name.rfind("blk.", 0) == 0) {
      size_t first_dot = tensor.name.find('.');
      size_t second_dot = tensor.name.find('.', first_dot + 1);
      size_t layer_index = std::stoul(
          tensor.name.substr(first_dot + 1, second_dot - first_dot - 1));
      std::string param_name = tensor.name.substr(second_dot + 1);

      if (layer_index < layers_.size()) {
        auto& layer = layers_[layer_index];
        if (param_name == "attn_norm.weight") {
          layer.attn_norm_weight = &tensor;
        } else if (param_name == "attn_q.weight") {
          layer.attn_q_weight = &tensor;
        } else if (param_name == "attn_k.weight") {
          layer.attn_k_weight = &tensor;
        } else if (param_name == "attn_v.weight") {
          layer.attn_v_weight = &tensor;
        } else if (param_name == "attn_output.weight") {
          layer.attn_output_weight = &tensor;
        } else if (param_name == "ffn_norm.weight") {
          layer.ffn_norm_weight = &tensor;
        } else if (param_name == "ffn_gate.weight") {
          layer.ffn_gate_weight = &tensor;
        } else if (param_name == "ffn_up.weight") {
          layer.ffn_up_weight = &tensor;
        } else if (param_name == "ffn_down.weight") {
          layer.ffn_down_weight = &tensor;
        } else if (param_name == "post_attention_norm.weight" ||
                   param_name == "attn_post_norm.weight") {
          layer.post_attention_norm_weight = &tensor;
        } else if (param_name == "post_ffw_norm.weight" ||
                   param_name == "ffn_post_norm.weight") {
          layer.post_ffw_norm_weight = &tensor;
        } else if (param_name == "attn_k_norm.weight") {
          layer.attn_k_norm_weight = &tensor;
        } else if (param_name == "attn_q_norm.weight") {
          layer.attn_q_norm_weight = &tensor;
        } else if (param_name == "out_scale.weight" ||
                   param_name == "layer_output_scale.weight") {
          layer.out_scale_weight = &tensor;
        } else if (param_name == "per_layer_inp_gate.weight" ||
                   param_name == "inp_gate.weight") {
          layer.per_layer_inp_gate_weight = &tensor;
        } else if (param_name == "per_layer_proj.weight" ||
                   param_name == "proj.weight") {
          layer.per_layer_proj_weight = &tensor;
        } else if (param_name == "per_layer_post_norm.weight" ||
                   param_name == "post_norm.weight") {
          layer.per_layer_post_norm_weight = &tensor;
        }
      }
    }
  }
}

tensor_2 Model::embed_tokens(const std::vector<int>& tokens) {
  const auto& token_embd_tensor = *token_embd_weight();
  size_t embedding_length = token_embd_tensor.shape[0];

  tensor_2 embeddings;
  embeddings.reserve(tokens.size());

  if (token_embd_tensor.tensor_type == (uint32_t)GGUFTensorType::F16) {
    // If we have loaded weights into memory, use them
    if (!token_embd_weight_f16_.empty()) {
      for (int token : tokens) {
        tensor_1 embedding_vector(embedding_length);
        size_t offset = token * embedding_length;
        for (size_t i = 0; i < embedding_length; ++i) {
          embedding_vector[i] = f16_to_f32(token_embd_weight_f16_[offset + i]);
        }
        embeddings.push_back(embedding_vector);
      }
    } else {
      // Fallback to reading from file (should not happen if constructor passed)
      std::vector<uint16_t> embedding_vector_f16(embedding_length);
      size_t row_size_bytes = embedding_length * sizeof(uint16_t);

      for (int token : tokens) {
        gguf_file_.read_tensor_data_region(
            token_embd_tensor, token * row_size_bytes,
            embedding_vector_f16.data(), row_size_bytes);

        tensor_1 embedding_vector(embedding_length);
        for (size_t i = 0; i < embedding_length; ++i) {
          embedding_vector[i] = f16_to_f32(embedding_vector_f16[i]);
        }
        embeddings.push_back(embedding_vector);
      }
    }
  } else if (token_embd_tensor.tensor_type == (uint32_t)GGUFTensorType::Q6_K) {
    const size_t bytes_per_block = sizeof(block_q6_K);
    const size_t blocks_per_row = embedding_length / QK_K;
    const size_t bytes_per_row = blocks_per_row * bytes_per_block;

    std::vector<uint8_t> embedding_row_q6k(bytes_per_row);

    for (int token : tokens) {
      gguf_file_.read_tensor_data_region(
          token_embd_tensor, token * bytes_per_row, embedding_row_q6k.data(),
          bytes_per_row);

      tensor_1 embedding_vector;
      dequantize_q6_k_row(embedding_vector, embedding_row_q6k.data(),
                          embedding_length);
      embeddings.push_back(embedding_vector);
    }
  } else if (token_embd_tensor.tensor_type == (uint32_t)GGUFTensorType::Q8_0) {
    const size_t bytes_per_block = sizeof(BlockQ8_0);
    const size_t blocks_per_row = embedding_length / 32;
    const size_t bytes_per_row = blocks_per_row * bytes_per_block;

    std::vector<uint8_t> embedding_row_q8_0(bytes_per_row);

    for (int token : tokens) {
      gguf_file_.read_tensor_data_region(
          token_embd_tensor, token * bytes_per_row, embedding_row_q8_0.data(),
          bytes_per_row);

      tensor_1 embedding_vector;
      dequantize_q8_0_row(embedding_vector, embedding_row_q8_0.data(),
                          embedding_length);
      embeddings.push_back(embedding_vector);
    }
  } else if (token_embd_tensor.tensor_type == (uint32_t)GGUFTensorType::Q5_0) {
    const size_t bytes_per_block = sizeof(block_q5_0);
    const size_t blocks_per_row = embedding_length / 32;
    const size_t bytes_per_row = blocks_per_row * bytes_per_block;

    std::vector<uint8_t> embedding_row_q5_0(bytes_per_row);

    for (int token : tokens) {
      gguf_file_.read_tensor_data_region(
          token_embd_tensor, token * bytes_per_row, embedding_row_q5_0.data(),
          bytes_per_row);

      tensor_1 embedding_vector;
      dequantize_q5_0_row(embedding_vector, embedding_row_q5_0.data(),
                          embedding_length);
      embeddings.push_back(embedding_vector);
    }
  } else {
    std::cerr << "Error: embed_tokens: Unsupported token embedding "
                 "tensor type: "
              << token_embd_tensor.tensor_type << std::endl;
    exit(1);
    // Handle other types later
  }
  return embeddings;
}

void Model::scale_embeddings(tensor_2& embeddings) {
  const float scaling =
      std::sqrt(static_cast<float>(hparams_.embedding_length));
  for (auto& embedding_vector : embeddings) {
    for (float& val : embedding_vector) {
      val *= scaling;
    }
  }
}

tensor_1 Model::run_norm(const tensor_1& input,
                         const TensorInfo* norm_weight_tensor) {
  tensor_1 norm_weight(norm_weight_tensor->shape[0]);
  gguf_file_.read_tensor_data(*norm_weight_tensor, norm_weight.data(),
                              norm_weight.size() * sizeof(float));
  tensor_1 normalized_x(input.size());
  rms_norm(normalized_x, input, hparams_.f_norm_rms_eps);
  VERBOSE(print_tensor(normalized_x, "norm"));
  tensor_1 norm_output(input.size());
  for (size_t j = 0; j < normalized_x.size(); ++j) {
    norm_output[j] = normalized_x[j] * norm_weight[j];
  }
  return norm_output;
}

tensor_2 Model::run_norm(const tensor_2& input,
                         const TensorInfo* norm_weight_tensor, int layer_id) {
  tensor_2 rms_norm_outputs;
  rms_norm_outputs.reserve(input.size());
  for (const auto& state : input) {
    tensor_1 normalized_x(state.size());
    rms_norm(normalized_x, state, hparams_.f_norm_rms_eps);
    rms_norm_outputs.push_back(normalized_x);
  }
  VERBOSE(print_tensor(rms_norm_outputs, "norm-" + std::to_string(layer_id)));

  tensor_1 norm_weight(norm_weight_tensor->shape[0]);
  gguf_file_.read_tensor_data(*norm_weight_tensor, norm_weight.data(),
                              norm_weight.size() * sizeof(float));

  tensor_2 final_outputs;
  final_outputs.reserve(input.size());
  for (const auto& normalized_x : rms_norm_outputs) {
    tensor_1 norm_output(normalized_x.size());
    for (size_t j = 0; j < normalized_x.size(); ++j) {
      norm_output[j] = normalized_x[j] * norm_weight[j];
    }
    final_outputs.push_back(norm_output);
  }
  return final_outputs;
}

tensor_3 Model::run_norm(const tensor_3& input,
                         const TensorInfo* norm_weight_tensor, int layer_id) {
  tensor_1 norm_weight(norm_weight_tensor->shape[0]);
  gguf_file_.read_tensor_data(*norm_weight_tensor, norm_weight.data(),
                              norm_weight.size() * sizeof(float));

  tensor_3 rms_norm_outputs;
  rms_norm_outputs.reserve(input.size());
  for (const auto& state_2d : input) {
    tensor_2 current_2d_norms;
    current_2d_norms.reserve(state_2d.size());
    for (const auto& state_1d : state_2d) {
      tensor_1 normalized_x(state_1d.size());
      rms_norm(normalized_x, state_1d, hparams_.f_norm_rms_eps);
      current_2d_norms.push_back(normalized_x);
    }
    rms_norm_outputs.push_back(current_2d_norms);
  }
  VERBOSE(print_tensor(rms_norm_outputs, "norm-" + std::to_string(layer_id)));

  tensor_3 final_outputs;
  final_outputs.reserve(input.size());
  for (const auto& state_2d : rms_norm_outputs) {
    tensor_2 current_2d_outputs;
    current_2d_outputs.reserve(state_2d.size());
    for (const auto& normalized_x : state_2d) {
      tensor_1 norm_output(normalized_x.size());
      for (size_t j = 0; j < normalized_x.size(); ++j) {
        norm_output[j] = normalized_x[j] * norm_weight[j];
      }
      current_2d_outputs.push_back(norm_output);
    }
    final_outputs.push_back(current_2d_outputs);
  }
  return final_outputs;
}

// Similar to
// https://github.com/ggml-org/llama.cpp/blob/e1f15b454fbadfddf8f1ec450bf6d390d9db7adb/src/llama-graph.cpp#L1777
// For building the graph. For actually running the attention, the llama.cpp
// reference is at:
// https://github.com/ggml-org/llama.cpp/blob/e1f15b454fbadfddf8f1ec450bf6d390d9db7adb/ggml/src/ggml-cpu/ops.cpp#L8333
tensor_2 Model::run_attn(KVCacheLayer& kv_cache,
                         const TensorInfo* output_weights,
                         const tensor_3& q_heads, const tensor_3& k_heads,
                         const tensor_3& v_heads, uint32_t n_head,
                         uint32_t n_head_kv, uint32_t n_embd_head,
                         int layer_index, int pos) {
  const uint32_t n_tokens = q_heads.size();
  const float max_bias = hparams_.f_max_alibi_bias;
  const float logit_softcap = hparams_.attn_soft_cap;

  // Concat k_heads and v_heads to kv_cache
  // Convert F32 heads to F16 for storage
  auto to_f16_3 = [](const tensor_3& src) {
    tensor_f16_3 dst;
    dst.reserve(src.size());
    for (const auto& vec2 : src) {
      tensor_f16_2 d2;
      d2.reserve(vec2.size());
      for (const auto& vec1 : vec2) {
        tensor_f16_1 d1;
        d1.reserve(vec1.size());
        for (float v : vec1) {
          d1.push_back(f32_to_f16(v));
        }
        d2.push_back(d1);
      }
      dst.push_back(d2);
    }
    return dst;
  };

  if (!k_heads.empty()) {
    tensor_f16_3 k_heads_f16 = to_f16_3(k_heads);
    tensor_f16_3 v_heads_f16 = to_f16_3(v_heads);

    if (kv_cache.k.empty()) {
      kv_cache.k = k_heads_f16;
      kv_cache.v = v_heads_f16;
    } else {
      kv_cache.k.insert(kv_cache.k.end(), k_heads_f16.begin(),
                        k_heads_f16.end());
      kv_cache.v.insert(kv_cache.v.end(), v_heads_f16.begin(),
                        v_heads_f16.end());
    }
  }

  tensor_2 kqv_out;
  kqv_out.reserve(n_tokens);
  for (uint32_t t = 0; t < n_tokens; ++t) {
    tensor_1 concatenated_head_results(n_head * n_embd_head, 0.0f);

    for (uint32_t h = 0; h < n_head; ++h) {
      const tensor_1& q_vec = q_heads[t][h];

      // Use FP16 for v_acc to match llama.cpp
      tensor_f16_1 v_acc(n_embd_head, f32_to_f16(0.0f));

      float s_acc = 0.0f;
      float max_score = -INFINITY;

      const uint32_t h_kv = h / (n_head / n_head_kv);

      float slope = 1.0f;
      if (max_bias > 0.0f) {
        const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));
        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
        slope = h < n_head_log2 ? powf(m0, h + 1)
                                : powf(m1, 2 * (h - n_head_log2) + 1);
      }

      for (uint32_t t_k = 0; t_k <= pos + t; ++t_k) {
        const tensor_f16_1& k_vec = kv_cache.k[t_k][h_kv];

        double score = 0.0f;
        for (uint32_t i = 0; i < n_embd_head; ++i) {
          // Downcast Q value to f16 to align with llama.cpp.
          uint16_t Q_q = f32_to_f16(q_vec[i]);
          score += (double)(f16_to_f32(k_vec[i]) * f16_to_f32(Q_q));
        }

        if (logit_softcap > 0.0f) {
          score = logit_softcap * tanhf(score / logit_softcap);
        }

        // ALiBi bias
        if (max_bias > 0.0f) {
          score += slope * (t_k - (pos + t));
        }

        const float prev_max_score = max_score;
        float score_exp;
        float prev_score_exp;
        if (score > prev_max_score) {
          max_score = score;
          score_exp = 1.0f;
          prev_score_exp = expf(prev_max_score - max_score);
          // V = V*expf(Mold - M)
          vec_scale_f16(v_acc, prev_score_exp);
        } else {
          score_exp = expf(score - max_score);
          prev_score_exp = 1.0f;
          // No scaling needed as prev_score_exp is 1.0
        }

        const tensor_f16_1& v_vec = kv_cache.v[t_k][h_kv];

        // V += v*expf(s - M)
        vec_mad_f16(v_acc, v_vec, score_exp);

        s_acc = s_acc * prev_score_exp + score_exp;
      }

      for (uint32_t i = 0; i < n_embd_head; ++i) {
        const float S_inv = s_acc == 0.0f ? 0.0f : 1.0f / s_acc;
        concatenated_head_results[h * n_embd_head + i] =
            f16_to_f32(v_acc[i]) * S_inv;
      }
    }
    kqv_out.push_back(concatenated_head_results);
  }
  VERBOSE(print_tensor(kqv_out, "kqv_out-" + std::to_string(layer_index)));

  tensor_2 all_attention_results;
  all_attention_results.reserve(n_tokens);
  for (const auto& concatenated_head_results : kqv_out) {
    tensor_1 token_result(output_weights->shape[1]);
    mat_vec_mul(token_result, *output_weights, gguf_file_,
                concatenated_head_results);
    all_attention_results.push_back(token_result);
  }
  VERBOSE(print_tensor(all_attention_results,
                       "attention results (node_30 for MUL_MAT)-" +
                           std::to_string(layer_index)));

  return all_attention_results;
}

tensor_3 Model::get_per_layer_inputs(const std::vector<int>& tokens) {
  if (!token_embd_per_layer_weight_) return {};

  const uint32_t n_embd_per_layer = hparams_.embedding_length_per_layer;
  const uint32_t n_layer = hparams_.block_count;
  const uint32_t n_tokens = tokens.size();

  // model.tok_embd_per_layer shape is [n_embd_per_layer * n_layer, vocab_size]
  // We need to extract rows and reshape.
  size_t row_size_elements = n_embd_per_layer * n_layer;
  size_t row_size_bytes = 0;
  if (token_embd_per_layer_weight_->tensor_type ==
      (uint32_t)GGUFTensorType::F16) {
    row_size_bytes = row_size_elements * sizeof(uint16_t);
  } else if (token_embd_per_layer_weight_->tensor_type ==
             (uint32_t)GGUFTensorType::Q6_K) {
    row_size_bytes = (row_size_elements / QK_K) * sizeof(block_q6_K);
  } else if (token_embd_per_layer_weight_->tensor_type ==
             (uint32_t)GGUFTensorType::Q4_K) {
    row_size_bytes = (row_size_elements / QK_K) * sizeof(block_q4_K);
  } else {
    std::cerr << "Error: get_per_layer_inputs: Unsupported tensor type: "
              << token_embd_per_layer_weight_->tensor_type << std::endl;
    exit(1);
  }

  tensor_3 inp_per_layer;  // [tokens][layer][embd_per_layer]
  inp_per_layer.reserve(n_tokens);

  std::vector<uint8_t> row_buf(row_size_bytes);
  float scale = std::sqrt((float)n_embd_per_layer);

  for (int token : tokens) {
    gguf_file_.read_tensor_data_region(*token_embd_per_layer_weight_,
                                       token * row_size_bytes, row_buf.data(),
                                       row_size_bytes);

    tensor_1 full_row;
    if (token_embd_per_layer_weight_->tensor_type ==
        (uint32_t)GGUFTensorType::F16) {
      full_row.resize(row_size_elements);
      const uint16_t* f16_ptr =
          reinterpret_cast<const uint16_t*>(row_buf.data());
      for (size_t i = 0; i < row_size_elements; ++i) {
        full_row[i] = f16_to_f32(f16_ptr[i]) * scale;
      }
    } else if (token_embd_per_layer_weight_->tensor_type ==
               (uint32_t)GGUFTensorType::Q6_K) {
      dequantize_q6_k_row(full_row, row_buf.data(), row_size_elements);
      for (float& val : full_row) val *= scale;
    } else if (token_embd_per_layer_weight_->tensor_type ==
               (uint32_t)GGUFTensorType::Q4_K) {
      dequantize_q4_k_row(full_row, row_buf.data(), row_size_elements);
      for (float& val : full_row) val *= scale;
    } else {
      std::cerr << "Error: get_per_layer_inputs: Unsupported tensor type: "
                << token_embd_per_layer_weight_->tensor_type << std::endl;
      exit(1);
    }

    tensor_2 layer_inputs;
    layer_inputs.reserve(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
      tensor_1 layer_embd(n_embd_per_layer);
      for (uint32_t i = 0; i < n_embd_per_layer; ++i) {
        layer_embd[i] = full_row[l * n_embd_per_layer + i];
      }
      layer_inputs.push_back(layer_embd);
    }
    inp_per_layer.push_back(layer_inputs);
  }

  return inp_per_layer;
}

tensor_3 Model::project_per_layer_inputs(const tensor_2& inputs_embeds,
                                         tensor_3& inp_per_layer) {
  if (!per_layer_model_proj_weight_) return inp_per_layer;

  const uint32_t n_tokens = inputs_embeds.size();
  const uint32_t n_embd = hparams_.embedding_length;
  const uint32_t n_embd_per_layer = hparams_.embedding_length_per_layer;
  const uint32_t n_layer = hparams_.block_count;

  const float per_layer_projection_scale = 1.0f / std::sqrt((float)n_embd);
  const float per_layer_input_scale = 1.0f / std::sqrt(2.0f);

  tensor_3 projected;  // [tokens][layer][embd_per_layer]
  projected.reserve(n_tokens);

  for (size_t t = 0; t < n_tokens; ++t) {
    tensor_1 proj_out(per_layer_model_proj_weight_->shape[1]);
    mat_vec_mul(proj_out, *per_layer_model_proj_weight_, gguf_file_,
                inputs_embeds[t]);

    // Scale
    for (float& val : proj_out) val *= per_layer_projection_scale;

    // Reshape proj_out [n_embd_per_layer * n_layer] to
    // [n_layer][n_embd_per_layer]
    tensor_2 layer_projs;
    layer_projs.reserve(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
      tensor_1 layer_proj(n_embd_per_layer);
      for (uint32_t i = 0; i < n_embd_per_layer; ++i) {
        layer_proj[i] = proj_out[l * n_embd_per_layer + i];
      }
      layer_projs.push_back(layer_proj);
    }
    projected.push_back(layer_projs);
  }

  // Now RMSNorm projected [tokens][layer][embd_per_layer] over embd_per_layer
  // per_layer_proj_norm is [n_embd_per_layer]? Wait, llama.cpp says -1 which
  // usually means shared over non-channel dims. In gemma4-iswa.cpp:
  // per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm,
  // nullptr, LLM_NORM_RMS, -1); This norm weight is [n_embd_per_layer].

  tensor_1 norm_weight(n_embd_per_layer);
  gguf_file_.read_tensor_data(*per_layer_proj_norm_weight_, norm_weight.data(),
                              n_embd_per_layer * sizeof(float));

  for (size_t t = 0; t < n_tokens; ++t) {
    for (size_t l = 0; l < n_layer; ++l) {
      tensor_1 normalized_x(n_embd_per_layer);
      rms_norm(normalized_x, projected[t][l], hparams_.f_norm_rms_eps);
      for (size_t i = 0; i < n_embd_per_layer; ++i) {
        float val =
            (normalized_x[i] * norm_weight[i] + inp_per_layer[t][l][i]) *
            per_layer_input_scale;
        inp_per_layer[t][l][i] = val;
      }
    }
  }

  return inp_per_layer;
}

tensor_2 Model::forward(const std::vector<int>& tokens, int pos) {
  LOG_VERBOSE("Starting forward pass.");

  // 1. Embedding lookup
  tensor_2 hidden_states = embed_tokens(tokens);
  VERBOSE(print_tensor(hidden_states, "imp_embed"));
  scale_embeddings(hidden_states);
  VERBOSE(print_tensor(hidden_states, "inp_scaled"));

  tensor_3 inp_per_layer;
  if (token_embd_per_layer_weight_) {
    inp_per_layer = get_per_layer_inputs(tokens);
    inp_per_layer = project_per_layer_inputs(hidden_states, inp_per_layer);
  }

  // Transformer blocks
  for (size_t i = 0; i < layers_.size(); ++i) {
    bool is_swa = false;
    if (i < hparams_.swa_layers.size()) {
      is_swa = hparams_.swa_layers[i];
    } else {
      size_t swa_n_pattern = 6;
      is_swa = i % swa_n_pattern < (swa_n_pattern - 1);
    }

    // Gemma will use a different freq_base depending on the layer.
    float this_rope_freq_base = is_swa ? 10000 : hparams_.rope_freq_base;
    auto& layer = layers_[i];

    tensor_2 normalized_states =
        run_norm(hidden_states, layer.attn_norm_weight, i);
    VERBOSE(print_tensor(normalized_states, "attn_norm-" + std::to_string(i)));

    uint32_t n_head = hparams_.attention_head_count;
    uint32_t n_head_kv = hparams_.attention_head_count_kv;
    uint32_t n_tokens = tokens.size();
    if (n_tokens == 0) {
      return {};
    }
    uint32_t n_embd_head_k =
        is_swa ? hparams_.n_embd_head_k_swa : hparams_.n_embd_head_k;
    uint32_t n_embd_head_v =
        is_swa ? hparams_.n_embd_head_v_swa : hparams_.n_embd_head_v;

    // Q
    tensor_2 q_vectors;
    for (const auto& normalized_x : normalized_states) {
      tensor_1 q(layer.attn_q_weight->shape[1]);
      mat_vec_mul(q, *layer.attn_q_weight, gguf_file_, normalized_x);
      q_vectors.push_back(q);
    }
    VERBOSE(print_tensor(q_vectors, "Qcur-" + std::to_string(i)));
    tensor_3 q_reshaped =
        reshape_3d(q_vectors, n_tokens, n_head, n_embd_head_k);
    VERBOSE(
        print_tensor(q_reshaped, "Qcur-" + std::to_string(i) + " (reshaped)"));
    tensor_3 q_cur = run_norm(q_reshaped, layer.attn_q_norm_weight, i);
    VERBOSE(print_tensor(q_cur, "Qcur_normed-" + std::to_string(i)));
    rope(q_cur, n_embd_head_k, this_rope_freq_base, hparams_.rope_freq_scale,
         pos);
    VERBOSE(print_tensor(q_cur, "Qcur-" + std::to_string(i) + " (post rope)"));
    scale(q_cur, hparams_.f_attention_scale);
    VERBOSE(
        print_tensor(q_cur, "node_9-" + std::to_string(i) + " (post scale)"));

    tensor_3 k_cur;
    tensor_3 v_heads;

    bool has_kv = true;
    if (hparams_.n_layer_kv_from_start >= 0) {
      has_kv = (int)i < hparams_.n_layer_kv_from_start;
    }

    if (has_kv) {
      // K
      tensor_2 k_vectors;
      for (const auto& normalized_x : normalized_states) {
        tensor_1 k(layer.attn_k_weight->shape[1]);
        mat_vec_mul(k, *layer.attn_k_weight, gguf_file_, normalized_x);
        k_vectors.push_back(k);
      }
      VERBOSE(print_tensor(k_vectors, "Kcur-" + std::to_string(i)));
      tensor_3 k_reshaped =
          reshape_3d(k_vectors, n_tokens, n_head_kv, n_embd_head_k);
      VERBOSE(print_tensor(k_reshaped,
                           "Kcur-" + std::to_string(i) + " (reshaped)"));
      k_cur = run_norm(k_reshaped, layer.attn_k_norm_weight, i);
      VERBOSE(print_tensor(k_cur, "Kcur_normed-" + std::to_string(i)));
      rope(k_cur, n_embd_head_k, this_rope_freq_base, hparams_.rope_freq_scale,
           pos);
      VERBOSE(
          print_tensor(k_cur, "Kcur-" + std::to_string(i) + " (post rope)"));

      // V
      tensor_2 v_vectors;
      for (const auto& normalized_x : normalized_states) {
        tensor_1 v(layer.attn_v_weight->shape[1]);
        mat_vec_mul(v, *layer.attn_v_weight, gguf_file_, normalized_x);
        v_vectors.push_back(v);
      }
      VERBOSE(print_tensor(v_vectors, "Vcur-" + std::to_string(i)));

      tensor_3 v_reshaped =
          reshape_3d(v_vectors, n_tokens, n_head_kv, n_embd_head_v);
      VERBOSE(print_tensor(v_reshaped,
                           "Vcur-" + std::to_string(i) + " (reshaped)"));

      if (hparams_.architecture == "gemma4") {
        // Gemma 4 RMSNorm on V (no weights)
        v_heads.reserve(n_tokens);
        for (const auto& v_token_heads : v_reshaped) {
          tensor_2 layer_normed_heads;
          layer_normed_heads.reserve(n_head_kv);
          for (const auto& v_head_vec : v_token_heads) {
            tensor_1 normalized_v(v_head_vec.size());
            rms_norm(normalized_v, v_head_vec, hparams_.f_norm_rms_eps);
            layer_normed_heads.push_back(normalized_v);
          }
          v_heads.push_back(layer_normed_heads);
        }
        VERBOSE(print_tensor(v_heads, "Vcur_normed-" + std::to_string(i)));
      } else {
        v_heads = v_reshaped;
      }
    }

    size_t src_il = i;
    if (!has_kv) {
      src_il = hparams_.n_layer_kv_from_start - (is_swa ? 2 : 1);
    }

    // Note: run_attn still needs dequantized output_weights for now
    // TODO: Optimize run_attn to use Q4_0 directly
    tensor_2 attention_results =
        run_attn(kv_cache_[src_il], layer.attn_output_weight, q_cur, k_cur,
                 v_heads, n_head, n_head_kv, n_embd_head_v, i, pos);

    if (layer.post_attention_norm_weight) {
      attention_results =
          run_norm(attention_results, layer.post_attention_norm_weight, i);
      VERBOSE(print_tensor(attention_results,
                           "attn_post_norm-" + std::to_string(i)));
    }

    for (size_t j = 0; j < hidden_states.size(); ++j) {
      for (size_t k = 0; k < hidden_states[j].size(); ++k) {
        hidden_states[j][k] += attention_results[j][k];
      }
    }
    VERBOSE(print_tensor(hidden_states, "sa_out-" + std::to_string(i)));

    tensor_2 normalized_states2 =
        run_norm(hidden_states, layer.ffn_norm_weight, i);
    VERBOSE(print_tensor(normalized_states2, "ffn_norm-" + std::to_string(i)));

    if (layer.ffn_gate_weight == nullptr || layer.ffn_up_weight == nullptr ||
        layer.ffn_down_weight == nullptr) {
      std::cerr << "Error: One of the FFN weights is null for layer " << i
                << std::endl;
      exit(1);
    }

    tensor_2 ffn_gate_outputs;
    ffn_gate_outputs.reserve(normalized_states2.size());
    tensor_2 ffn_up_outputs;
    ffn_up_outputs.reserve(normalized_states2.size());
    for (const auto& normalized_x2 : normalized_states2) {
      tensor_1 ffn_gate_output(layer.ffn_gate_weight->shape[1]);
      tensor_1 ffn_up_output(layer.ffn_up_weight->shape[1]);
      mat_vec_mul(ffn_gate_output, *layer.ffn_gate_weight, gguf_file_,
                  normalized_x2);
      mat_vec_mul(ffn_up_output, *layer.ffn_up_weight, gguf_file_,
                  normalized_x2);
      ffn_gate_outputs.push_back(ffn_gate_output);
      ffn_up_outputs.push_back(ffn_up_output);
    }
    VERBOSE(print_tensor(ffn_gate_outputs, "ffn_gate-" + std::to_string(i)));
    VERBOSE(print_tensor(ffn_up_outputs, "ffn_up-" + std::to_string(i)));

    tensor_2 ffn_gate_up_combined;
    ffn_gate_up_combined.reserve(normalized_states2.size());
    for (size_t token_idx = 0; token_idx < normalized_states2.size();
         ++token_idx) {
      const tensor_1& ffn_gate_output = ffn_gate_outputs[token_idx];
      const tensor_1& ffn_up_output = ffn_up_outputs[token_idx];

      tensor_1 ffn_hidden(ffn_gate_output.size());
      for (size_t j = 0; j < ffn_hidden.size(); ++j) {
        float x = ffn_gate_output[j];
        float gelu_x =
            0.5f * x *
            (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        ffn_hidden[j] = gelu_x * ffn_up_output[j];
      }
      ffn_gate_up_combined.push_back(ffn_hidden);
    }
    VERBOSE(
        print_tensor(ffn_gate_up_combined, "ffn_geglu-" + std::to_string(i)));

    tensor_2 ffn_outputs;
    ffn_outputs.reserve(normalized_states2.size());
    for (const auto& ffn_hidden_val : ffn_gate_up_combined) {
      tensor_1 ffn_output(layer.ffn_down_weight->shape[1]);
      mat_vec_mul(ffn_output, *layer.ffn_down_weight, gguf_file_,
                  ffn_hidden_val);
      ffn_outputs.push_back(ffn_output);
    }
    VERBOSE(print_tensor(ffn_outputs, "ffn_out-" + std::to_string(i)));

    if (layer.post_ffw_norm_weight) {
      ffn_outputs = run_norm(ffn_outputs, layer.post_ffw_norm_weight, i);
      VERBOSE(print_tensor(ffn_outputs, "ffn_post_norm-" + std::to_string(i)));
    }

    for (size_t j = 0; j < hidden_states.size(); ++j) {
      for (size_t k = 0; k < hidden_states[j].size(); ++k) {
        hidden_states[j][k] += ffn_outputs[j][k];
      }
    }

    // Per-layer embedding
    if (!inp_per_layer.empty()) {
      VERBOSE(print_tensor(hidden_states, "pe_in-" + std::to_string(i)));

      tensor_1 norm_weight(layer.per_layer_post_norm_weight->shape[0]);
      gguf_file_.read_tensor_data(*layer.per_layer_post_norm_weight,
                                  norm_weight.data(),
                                  norm_weight.size() * sizeof(float));

      for (size_t token_idx = 0; token_idx < n_tokens; ++token_idx) {
        tensor_1 gate_out(layer.per_layer_inp_gate_weight->shape[1]);
        mat_vec_mul(gate_out, *layer.per_layer_inp_gate_weight, gguf_file_,
                    hidden_states[token_idx]);

        // GELU
        for (float& x : gate_out) {
          x = 0.5f * x *
              (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        }

        // Mul with inp_per_layer
        for (size_t j = 0; j < gate_out.size(); ++j) {
          gate_out[j] *= inp_per_layer[token_idx][i][j];
        }

        // Proj
        tensor_1 proj_out(layer.per_layer_proj_weight->shape[1]);
        mat_vec_mul(proj_out, *layer.per_layer_proj_weight, gguf_file_,
                    gate_out);

        // Post norm
        tensor_1 normalized_proj(proj_out.size());
        rms_norm(normalized_proj, proj_out, hparams_.f_norm_rms_eps);

        for (size_t j = 0; j < hidden_states[token_idx].size(); ++j) {
          hidden_states[token_idx][j] += normalized_proj[j] * norm_weight[j];
        }
      }
      VERBOSE(print_tensor(hidden_states,
                           "per_layer_embd_out-" + std::to_string(i)));
    }

    // layer_scalar (out_scale)
    if (layer.out_scale_weight) {
      float out_scale;
      gguf_file_.read_tensor_data(*layer.out_scale_weight, &out_scale,
                                  sizeof(float));
      for (auto& state : hidden_states) {
        for (float& val : state) val *= out_scale;
      }
      VERBOSE(print_tensor(hidden_states, "out_scaled-" + std::to_string(i)));
    }

    VERBOSE(print_tensor(hidden_states, "l_out-" + std::to_string(i)));
  }

  // Final RMSNorm
  tensor_1& last_hidden_state = hidden_states.back();

  tensor_1 final_normalized_x =
      run_norm(last_hidden_state, output_norm_weight());

  VERBOSE(print_tensor(final_normalized_x, "result_norm"));

  // Output logits

  // Output logits
  const auto& token_embd_weight_tensor = *token_embd_weight();
  size_t embedding_length = token_embd_weight_tensor.shape[0];
  size_t vocab_size = token_embd_weight_tensor.shape[1];

  tensor_1 logits(vocab_size);

  if (!token_embd_weight_f16_.empty()) {
    mat_vec_mul_fp16(logits, token_embd_weight_f16_, final_normalized_x,
                     vocab_size, embedding_length);
  } else if (token_embd_weight_tensor.tensor_type ==
             (uint32_t)GGUFTensorType::F16) {
    std::vector<uint16_t> embedding_row_f16(embedding_length);
    size_t row_size_bytes = embedding_length * sizeof(uint16_t);

    for (size_t token = 0; token < vocab_size; ++token) {
      gguf_file_.read_tensor_data_region(
          token_embd_weight_tensor, token * row_size_bytes,
          embedding_row_f16.data(), row_size_bytes);

      float logit = 0.0f;
      for (size_t i = 0; i < embedding_length; ++i) {
        float weight = f16_to_f32(embedding_row_f16[i]);
        logit += weight * final_normalized_x[i];
      }
      logits[token] = logit;
    }
  } else if (token_embd_weight_tensor.tensor_type ==
                 (uint32_t)GGUFTensorType::Q6_K ||
             token_embd_weight_tensor.tensor_type ==
                 (uint32_t)GGUFTensorType::Q4_K ||
             token_embd_weight_tensor.tensor_type ==
                 (uint32_t)GGUFTensorType::Q8_0 ||
             token_embd_weight_tensor.tensor_type ==
                 (uint32_t)GGUFTensorType::Q5_0) {
    mat_vec_mul(logits, token_embd_weight_tensor, gguf_file_,
                final_normalized_x);
  } else {
    std::cerr
        << "Error: forward: Unsupported token embedding tensor type for logits."
        << std::endl;
    exit(1);
  }

  if (hparams_.final_logit_softcap > 0.0f) {
    for (float& logit : logits) {
      logit = hparams_.final_logit_softcap *
              tanhf(logit / hparams_.final_logit_softcap);
    }
  }

  tensor_2 result;
  result.push_back(logits);

  VERBOSE(print_tensor(result, "result_output"));

  return result;
}

void Model::load_vocabulary() {
  const auto& metadata = gguf_file_.get_metadata();

  const auto& tokens_value = metadata.at("tokenizer.ggml.tokens");
  id_to_token.reserve(tokens_value.arr.size());
  for (const auto& token : tokens_value.arr) {
    id_to_token.push_back(token.str);
  }

  for (size_t i = 0; i < id_to_token.size(); ++i) {
    token_to_id[id_to_token[i]] = i;
  }

  if (metadata.count("tokenizer.ggml.bos_token_id")) {
    bos_token_id = metadata.at("tokenizer.ggml.bos_token_id").scalar.u32;
  } else {
    auto it = token_to_id.find("<bos>");
    if (it != token_to_id.end()) {
      bos_token_id = it->second;
    } else if (token_to_id.size() > 2) {
      bos_token_id = 2;  // Default for Gemma
    }
  }

  if (metadata.count("tokenizer.ggml.unk_token_id")) {
    unk_token_id = metadata.at("tokenizer.ggml.unk_token_id").scalar.u32;
  } else {
    auto it = token_to_id.find("<unk>");
    if (it != token_to_id.end()) {
      unk_token_id = it->second;
    }
  }

  for (const auto& token_str : id_to_token) {
    if (token_str.length() > max_token_len) {
      max_token_len = token_str.length();
    }
  }
}

std::vector<int> Model::tokenize(const std::string& prompt,
                                 bool apply_chat_template,
                                 bool* out_prefilled_thinking) {
  if (out_prefilled_thinking) {
    *out_prefilled_thinking = false;
  }
  std::vector<int> tokens;
  std::string processed_prompt;
  // Warning: Ideally we should be running the template in
  // tokenizer.chat_template to support more models. Just use this hardcoded
  // template for now.
  if (apply_chat_template) {
    if (hparams_.architecture == "gemma4") {
      bool add_bos = true;
      const auto& metadata = gguf_file_.get_metadata();
      if (metadata.count("tokenizer.ggml.add_bos_token")) {
        add_bos = metadata.at("tokenizer.ggml.add_bos_token").scalar.b;
      }
      if (add_bos && bos_token_id != -1) {
        tokens.push_back(bos_token_id);
      }
      processed_prompt =
          "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n<|channel>thought";
      if (out_prefilled_thinking) {
        *out_prefilled_thinking = true;
      }
    } else {
      if (bos_token_id != -1) {
        tokens.push_back(bos_token_id);
      }
      processed_prompt = "<start_of_turn>user\n" + prompt +
                         "<end_of_turn>\n<start_of_turn>model\n";
    }
  } else {
    if (hparams_.architecture == "gemma4") {
      processed_prompt = prompt;
    } else {
      if (bos_token_id != -1) {
        tokens.push_back(bos_token_id);
      }
      processed_prompt = " " + prompt;
    }
  }

  // Replace ASCII spaces with the UTF-8 "▁" (U+2581).
  // std::replace does not accept replacing a char with a multi-byte string,
  // so use find/replace loop instead.
  if (processed_prompt.find(' ') != std::string::npos) {
    const std::string repl = u8"\u2581";
    size_t pos = 0;
    while ((pos = processed_prompt.find(' ', pos)) != std::string::npos) {
      processed_prompt.replace(pos, 1, repl);
      pos += repl.size();
    }
  }
  LOG_VERBOSE("Processed prompt: " << processed_prompt);

  size_t i = 0;
  while (i < processed_prompt.length()) {
    bool found = false;
    size_t longest_match = 0;
    int best_token_id = -1;

    for (size_t len = 1;
         len <= max_token_len && i + len <= processed_prompt.length(); ++len) {
      std::string sub = processed_prompt.substr(i, len);
      auto it = token_to_id.find(sub);
      if (it != token_to_id.end()) {
        if (len > longest_match) {
          longest_match = len;
          best_token_id = it->second;
        }
      }
    }

    if (best_token_id != -1) {
      tokens.push_back(best_token_id);
      i += longest_match;
      found = true;
    }

    if (!found) {
      if (unk_token_id != -1) {
        tokens.push_back(unk_token_id);
      }
      i++;
    }
  }

  // For each token, print its string representation for debugging
  for (int token_id : tokens) {
    auto it = id_to_token.begin() + token_id;
    if (it != id_to_token.end()) {
      LOG_VERBOSE("Token ID: " << token_id << " String: \"" << *it << "\"");
    } else {
      LOG_VERBOSE("Token ID: " << token_id << " String: <unknown>");
    }
  }

  return tokens;
}

void Model::reset_kv_cache() {
  for (auto& layer : kv_cache_) {
    layer.k.clear();
    layer.v.clear();
  }
}

std::vector<int> Model::tokenize_chat(const std::vector<ChatMessage>& messages,
                                      bool enable_thinking,
                                      bool* out_prefilled_thinking) {
  if (out_prefilled_thinking) {
    *out_prefilled_thinking = false;
  }

  const auto& metadata = gguf_file_.get_metadata();

  // Look up BOS/EOS token strings.
  std::string bos_str =
      (bos_token_id >= 0 && bos_token_id < (int)id_to_token.size())
          ? id_to_token[bos_token_id]
          : "";
  std::string eos_str;
  if (metadata.count("tokenizer.ggml.eos_token_id")) {
    int eos_id = (int)metadata.at("tokenizer.ggml.eos_token_id").scalar.u32;
    if (eos_id >= 0 && eos_id < (int)id_to_token.size()) {
      eos_str = id_to_token[eos_id];
    }
  }

  if (metadata.count("tokenizer.chat_template")) {
    // Render the model's own chat template via inja (C++ Jinja2).
    const std::string& tmpl_src = metadata.at("tokenizer.chat_template").str;

    nlohmann::json msgs_json = nlohmann::json::array();
    for (const auto& msg : messages) {
      msgs_json.push_back({{"role", msg.role}, {"content", msg.content}});
    }
    nlohmann::json data = {
        {"messages", msgs_json},         {"bos_token", bos_str},
        {"eos_token", eos_str},          {"enable_thinking", enable_thinking},
        {"add_generation_prompt", true}, {"tools", nlohmann::json::array()},
    };

    try {
      inja::Environment env;
      std::string rendered = env.render(tmpl_src, data);
      LOG_VERBOSE("Rendered chat template:\n" << rendered);

      // Detect whether the rendered output ends with a thinking pre-fill token.
      const std::string think_tag = "<|channel>thought";
      if (out_prefilled_thinking && rendered.size() >= think_tag.size() &&
          rendered.compare(rendered.size() - think_tag.size(), think_tag.size(),
                           think_tag) == 0) {
        *out_prefilled_thinking = true;
      }

      return tokenize(rendered, /*apply_chat_template=*/false,
                      /*out_prefilled_thinking=*/nullptr);
    } catch (const std::exception& e) {
      // inja doesn't support every Jinja2 extension. Fall through to the
      // hardcoded implementation below.
      LOG_VERBOSE("Chat template rendering failed ("
                  << e.what() << "), using hardcoded template.");
    }
  }

  // Fallback: hardcoded templates for known architectures.
  std::string formatted;
  if (hparams_.architecture == "gemma4") {
    bool add_bos = true;
    if (metadata.count("tokenizer.ggml.add_bos_token")) {
      add_bos = metadata.at("tokenizer.ggml.add_bos_token").scalar.b;
    }
    if (add_bos && !bos_str.empty()) {
      formatted += bos_str;
    }
    for (const auto& msg : messages) {
      const std::string role = (msg.role == "assistant") ? "model" : msg.role;
      formatted += "<|turn>" + role + "\n" + msg.content + "<turn|>\n";
    }
    formatted += "<|turn>model\n";
    if (enable_thinking) {
      formatted += "<|channel>thought";
      if (out_prefilled_thinking) *out_prefilled_thinking = true;
    }
  } else {
    if (!bos_str.empty()) formatted += bos_str;
    for (const auto& msg : messages) {
      if (msg.role == "user") {
        formatted += "<start_of_turn>user\n" + msg.content + "<end_of_turn>\n";
      } else if (msg.role == "assistant") {
        formatted += "<start_of_turn>model\n" + msg.content + "<end_of_turn>\n";
      }
    }
    formatted += "<start_of_turn>model\n";
  }
  return tokenize(formatted, /*apply_chat_template=*/false,
                  /*out_prefilled_thinking=*/nullptr);
}