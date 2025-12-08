#include "model.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "tensor.h"

Model::Model(GGUFFile& gguf_file) : gguf_file_(gguf_file) {
  load_hparams(gguf_file);
  layers_.resize(hparams_.block_count);
  kv_cache_.resize(hparams_.block_count);
  map_tensors(gguf_file);
  load_vocabulary();

  if (token_embd_weight_->tensor_type == (uint32_t)GGUFTensorType::F16) {
    LOG_VERBOSE("Pre-converting token embedding weights to F32...");
    const size_t num_elements =
        token_embd_weight_->shape[0] * token_embd_weight_->shape[1];
    std::vector<uint16_t> token_embd_weight_f16(num_elements);
    gguf_file_.read_tensor_data(*token_embd_weight_,
                                token_embd_weight_f16.data(),
                                num_elements * sizeof(uint16_t));

    token_embd_weight_f32_.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      token_embd_weight_f32_[i] = f16_to_f32(token_embd_weight_f16[i]);
    }
    LOG_VERBOSE("Conversion done.");
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

  hparams_.block_count = get_key("gemma3.block_count")->scalar.u32;
  hparams_.embedding_length = get_key("gemma3.embedding_length")->scalar.u32;
  hparams_.feed_forward_length =
      get_key("gemma3.feed_forward_length")->scalar.u32;
  hparams_.attention_head_count =
      get_key("gemma3.attention.head_count")->scalar.u32;
  hparams_.attention_head_count_kv =
      get_key("gemma3.attention.head_count_kv")->scalar.u32;
  hparams_.f_norm_rms_eps =
      get_key("gemma3.attention.layer_norm_rms_epsilon")->scalar.f32;
  hparams_.rope_freq_base = get_key("gemma3.rope.freq_base")->scalar.f32;
  // Hardcoding this to 1, since it seems possible that that llama.cpp ignores
  // the gemma3.rope.scaling.factor of 8 on the larger Gemma model because the
  // rope_freq_base is already set to 1,000,000, which inherently handles the
  // extended context. Forcing the scaling factor to 1.0 which is a bit of a
  // hack, but it certainly helps with getting more accurate results.
  hparams_.rope_freq_scale = 1.0f;
  hparams_.n_embd_head_k =
      hparams_.embedding_length / hparams_.attention_head_count;
  const auto* attention_key_length_value =
      get_key("gemma3.attention.key_length", false);
  if (attention_key_length_value) {
    hparams_.n_embd_head_k = attention_key_length_value->scalar.u32;
  }
  hparams_.f_attention_scale = 1.0f / std::sqrt(float(hparams_.n_embd_head_k));

  const auto* max_alibi_bias_value =
      get_key("gemma3.attention.max_alibi_bias", false);
  hparams_.f_max_alibi_bias =
      max_alibi_bias_value ? max_alibi_bias_value->scalar.f32 : 0.0f;

  const auto* attn_soft_cap_value =
      get_key("gemma3.attention.logit_softcapping", false);
  hparams_.attn_soft_cap =
      attn_soft_cap_value ? attn_soft_cap_value->scalar.f32 : 0.0f;
}

void Model::map_tensors(GGUFFile& gguf_file) {
  auto& tensor_infos =
      const_cast<std::vector<TensorInfo>&>(gguf_file.get_tensor_infos());

  for (auto& tensor : tensor_infos) {
    if (tensor.name == "token_embd.weight") {
      token_embd_weight_ = &tensor;
    } else if (tensor.name == "output_norm.weight") {
      output_norm_weight_ = &tensor;
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
        } else if (param_name == "post_attention_norm.weight") {
          layer.post_attention_norm_weight = &tensor;
        } else if (param_name == "post_ffw_norm.weight") {
          layer.post_ffw_norm_weight = &tensor;
        } else if (param_name == "attn_k_norm.weight") {
          layer.attn_k_norm_weight = &tensor;
        } else if (param_name == "attn_q_norm.weight") {
          layer.attn_q_norm_weight = &tensor;
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
  } else {
    std::cerr << "Error: embed_tokens: Unsupported token embedding "
                 "tensor type."
              << std::endl;
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
                         const TensorInfo* norm_weight_tensor, int layer_id) {
  tensor_1 norm_weight(norm_weight_tensor->shape[0]);
  gguf_file_.read_tensor_data(*norm_weight_tensor, norm_weight.data(),
                              norm_weight.size() * sizeof(float));
  VERBOSE(
      print_tensor(norm_weight, "1d norm weight-" + std::to_string(layer_id)));
  tensor_1 normalized_x(input.size());
  rms_norm(normalized_x, input, hparams_.f_norm_rms_eps);
  VERBOSE(print_tensor(normalized_x, "norm-" + std::to_string(layer_id)));
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
  if (kv_cache.k.empty()) {
    kv_cache.k = k_heads;
    kv_cache.v = v_heads;
  } else {
    kv_cache.k.insert(kv_cache.k.end(), k_heads.begin(), k_heads.end());
    kv_cache.v.insert(kv_cache.v.end(), v_heads.begin(), v_heads.end());
  }

  tensor_2 kqv_out;
  kqv_out.reserve(n_tokens);
  for (uint32_t t = 0; t < n_tokens; ++t) {
    tensor_1 concatenated_head_results(n_head * n_embd_head, 0.0f);

    for (uint32_t h = 0; h < n_head; ++h) {
      const tensor_1& q_vec = q_heads[t][h];
      tensor_1 v_acc(n_embd_head, 0.0f);
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
        const tensor_1& k_vec = kv_cache.k[t_k][h_kv];

        float score = 0.0f;
        for (uint32_t i = 0; i < n_embd_head; ++i) {
          score += q_vec[i] * k_vec[i];
        }

        if (logit_softcap > 0.0f) {
          score = logit_softcap * tanhf(score / logit_softcap);
        }

        // ALiBi bias
        if (max_bias > 0.0f) {
          score += slope * (t_k - (pos + t));
        }

        const float prev_max_score = max_score;
        if (score > max_score) {
          max_score = score;
        }

        const float score_exp = expf(score - max_score);
        const float prev_score_exp = expf(prev_max_score - max_score);

        const tensor_1& v_vec = kv_cache.v[t_k][h_kv];

        for (uint32_t i = 0; i < n_embd_head; ++i) {
          v_acc[i] = v_acc[i] * prev_score_exp + v_vec[i] * score_exp;
        }
        s_acc = s_acc * prev_score_exp + score_exp;
      }

      for (uint32_t i = 0; i < n_embd_head; ++i) {
        if (s_acc != 0.0f) {
          concatenated_head_results[h * n_embd_head + i] = v_acc[i] / s_acc;
        }
      }
    }
    kqv_out.push_back(concatenated_head_results);
  }
  VERBOSE(print_tensor(kqv_out, "kqv_out-" + std::to_string(layer_index)));

  tensor_2 all_attention_results;
  all_attention_results.reserve(n_tokens);
  for (const auto& concatenated_head_results : kqv_out) {
    tensor_1 token_result(output_weights->shape[1]);
    mat_vec_mul_q4_0(token_result, *output_weights, gguf_file_,
                     concatenated_head_results);
    all_attention_results.push_back(token_result);
  }
  VERBOSE(print_tensor(all_attention_results,
                       "attention results (node_30 for MUL_MAT)-" +
                           std::to_string(layer_index)));

  return all_attention_results;
}

tensor_2 Model::forward(const std::vector<int>& tokens, int pos) {
  LOG_VERBOSE("Starting forward pass.");

  // 1. Embedding lookup
  tensor_2 hidden_states = embed_tokens(tokens);
  VERBOSE(print_tensor(hidden_states, "imp_embed"));
  scale_embeddings(hidden_states);
  VERBOSE(print_tensor(hidden_states, "inp_scaled"));

  // Transformer blocks
  for (size_t i = 0; i < layers_.size(); ++i) {
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
    uint32_t n_embd_head = hparams_.n_embd_head_k;

    // Q
    tensor_2 q_vectors;
    for (const auto& normalized_x : normalized_states) {
      tensor_1 q(layer.attn_q_weight->shape[1]);
      mat_vec_mul_q4_0(q, *layer.attn_q_weight, gguf_file_, normalized_x);
      q_vectors.push_back(q);
    }
    VERBOSE(print_tensor(q_vectors, "Qcur-" + std::to_string(i)));
    tensor_3 q_reshaped = reshape_3d(q_vectors, n_tokens, n_head, n_embd_head);
    VERBOSE(
        print_tensor(q_reshaped, "Qcur-" + std::to_string(i) + " (reshaped)"));
    tensor_3 q_cur = run_norm(q_reshaped, layer.attn_q_norm_weight, i);
    VERBOSE(print_tensor(q_cur, "Qcur_normed-" + std::to_string(i)));
    rope(q_cur, n_embd_head, hparams_.rope_freq_base, hparams_.rope_freq_scale,
         pos);
    VERBOSE(print_tensor(q_cur, "Qcur-" + std::to_string(i) + " (post rope)"));
    scale(q_cur, hparams_.f_attention_scale);
    VERBOSE(
        print_tensor(q_cur, "node_9-" + std::to_string(i) + " (post scale)"));

    // K
    tensor_2 k_vectors;
    for (const auto& normalized_x : normalized_states) {
      tensor_1 k(layer.attn_k_weight->shape[1]);
      mat_vec_mul_q4_0(k, *layer.attn_k_weight, gguf_file_, normalized_x);
      k_vectors.push_back(k);
    }
    VERBOSE(print_tensor(k_vectors, "Kcur-" + std::to_string(i)));
    tensor_3 k_reshaped =
        reshape_3d(k_vectors, n_tokens, n_head_kv, n_embd_head);
    VERBOSE(
        print_tensor(k_reshaped, "Kcur-" + std::to_string(i) + " (reshaped)"));
    tensor_3 k_cur = run_norm(k_reshaped, layer.attn_k_norm_weight, i);
    VERBOSE(print_tensor(k_cur, "Kcur_normed-" + std::to_string(i)));
    rope(k_cur, n_embd_head, hparams_.rope_freq_base, hparams_.rope_freq_scale,
         pos);
    VERBOSE(print_tensor(k_cur, "Kcur-" + std::to_string(i) + " (post rope)"));

    // V
    tensor_2 v_vectors;
    for (const auto& normalized_x : normalized_states) {
      tensor_1 v(layer.attn_v_weight->shape[1]);
      mat_vec_mul_q4_0(v, *layer.attn_v_weight, gguf_file_, normalized_x);
      v_vectors.push_back(v);
    }
    VERBOSE(print_tensor(v_vectors, "Vcur-" + std::to_string(i)));
    tensor_3 v_reshaped =
        reshape_3d(v_vectors, n_tokens, n_head_kv, n_embd_head);
    VERBOSE(
        print_tensor(v_reshaped, "Vcur-" + std::to_string(i) + " (reshaped)"));

    // Looks mostly accurate even up to initial {Q,K,V}cur calculations, except
    // that Q,K,V seem to have some slight differences, whereas attn_norm was
    // exact. So, I suspect it's because llama.cpp might be doing multiplication
    // in float32 but we're doing float only. I've determined that the
    // difference in value between this and llama.cpp is because llama.cpp casts
    // the float32 attn_norm to float8 before doing the matrix multiply.
    // Specifically, at this line:
    // https://github.com/ggml-org/llama.cpp/blob/7e994168b1ccc12337ba8de939c4fd466107c1fb/ggml/src/ggml-cpu/repack.cpp#L1660
    // Please see all the debugging notes at
    // ./random_coding_practice/tmp/llama_cpp_matrix_debugging_changes.diff For
    // now, my plan is to ignore this.

    // Note: run_attn still needs dequantized output_weights for now
    // TODO: Optimize run_attn to use Q4_0 directly
    tensor_2 attention_results =
        run_attn(kv_cache_[i], layer.attn_output_weight, q_cur, k_cur,
                 v_reshaped, n_head, n_head_kv, n_embd_head, i, pos);

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
      mat_vec_mul_q4_0(ffn_gate_output, *layer.ffn_gate_weight, gguf_file_,
                       normalized_x2);
      mat_vec_mul_q4_0(ffn_up_output, *layer.ffn_up_weight, gguf_file_,
                       normalized_x2);
      ffn_gate_outputs.push_back(ffn_gate_output);
      ffn_up_outputs.push_back(ffn_up_output);
    }
    VERBOSE(print_tensor(ffn_gate_outputs, "ffn_gate-" + std::to_string(i)));
    VERBOSE(print_tensor(ffn_up_outputs, "ffn_up-" + std::to_string(i)));

    tensor_2 all_ffn_hidden_outputs;
    all_ffn_hidden_outputs.reserve(normalized_states2.size());
    for (size_t token_idx = 0; token_idx < normalized_states2.size();
         ++token_idx) {
      tensor_1& ffn_gate_output = ffn_gate_outputs[token_idx];
      const tensor_1& ffn_up_output = ffn_up_outputs[token_idx];

      for (size_t j = 0; j < ffn_gate_output.size(); ++j) {
        ffn_gate_output[j] =
            ffn_gate_output[j] * (1.0f / (1.0f + expf(-ffn_gate_output[j])));
      }

      tensor_1 ffn_hidden(ffn_gate_output.size());
      for (size_t j = 0; j < ffn_hidden.size(); ++j) {
        ffn_hidden[j] = ffn_gate_output[j] * ffn_up_output[j];
      }
      all_ffn_hidden_outputs.push_back(ffn_hidden);
    }
    VERBOSE(
        print_tensor(all_ffn_hidden_outputs, "ffn_geglu-" + std::to_string(i)));

    tensor_2 ffn_outputs;
    ffn_outputs.reserve(normalized_states2.size());
    for (const auto& ffn_hidden_val : all_ffn_hidden_outputs) {
      tensor_1 ffn_output(layer.ffn_down_weight->shape[1]);
      mat_vec_mul_q4_0(ffn_output, *layer.ffn_down_weight, gguf_file_,
                       ffn_hidden_val);
      ffn_outputs.push_back(ffn_output);
    }
    VERBOSE(print_tensor(ffn_outputs, "ffn_out-" + std::to_string(i)));

    if (layer.post_ffw_norm_weight) {
      ffn_outputs = run_norm(ffn_outputs, layer.post_ffw_norm_weight, i);
    }
    VERBOSE(print_tensor(ffn_outputs, "ffn_post_norm-" + std::to_string(i)));

    for (size_t j = 0; j < hidden_states.size(); ++j) {
      for (size_t k = 0; k < hidden_states[j].size(); ++k) {
        hidden_states[j][k] += ffn_outputs[j][k];
      }
    }

    VERBOSE(print_tensor(hidden_states, "l_out-" + std::to_string(i)));
  }
  LOG_VERBOSE(" done.");

  // Final RMSNorm
  LOG_VERBOSE("Final RMSNorm...");
  tensor_1& last_hidden_state = hidden_states.back();
  VERBOSE(print_tensor(last_hidden_state, "last_hidden_state"));

  tensor_1 final_normalized_x =
      run_norm(last_hidden_state, output_norm_weight(), -1);

  VERBOSE(print_tensor(final_normalized_x, "result_norm"));

  // Output logits
  const auto& token_embd_weight_tensor = *token_embd_weight();
  size_t embedding_length = token_embd_weight_tensor.shape[0];
  size_t vocab_size = token_embd_weight_tensor.shape[1];

  tensor_1 logits(vocab_size);
  LOG_VERBOSE("Calculating logits...");

  if (!token_embd_weight_f32_.empty()) {
    mat_vec_mul(logits, token_embd_weight_f32_, final_normalized_x);
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
  } else {
    std::cerr
        << "Error: forward: Unsupported token embedding tensor type for logits."
        << std::endl;
    exit(1);
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
                                 bool apply_chat_template) {
  std::vector<int> tokens;
  if (bos_token_id != -1) {
    tokens.push_back(bos_token_id);
  } else {
    std::cerr << "Warning: BOS token not found. Not adding BOS token."
              << std::endl;
  }

  std::string processed_prompt;
  // Warning: Ideally we should be running the template in
  // tokenizer.chat_template to support more models. Just use this hardcoded
  // template for now.
  if (apply_chat_template) {
    processed_prompt = "<start_of_turn>user\n" + prompt +
                       "<end_of_turn>\n<start_of_turn>model\n";
  } else {
    processed_prompt = " " + prompt;
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