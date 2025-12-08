#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "gguf.h"
#include "ops.h"
#include "tensor.h"

/**
 * @brief Represents a layer in the Key-Value cache.
 */
struct KVCacheLayer {
  tensor_3 k;
  tensor_3 v;
};
using KVCache = std::vector<KVCacheLayer>;

/**
 * @brief Hyperparameters for the model.
 */
struct ModelHParams {
  uint32_t block_count;
  uint32_t embedding_length;
  uint32_t feed_forward_length;
  uint32_t attention_head_count;
  uint32_t attention_head_count_kv;
  double f_norm_rms_eps;
  float rope_freq_base;
  float rope_freq_scale;
  uint32_t n_embd_head_k;
  float f_attention_scale;
  float f_max_alibi_bias;
  float attn_soft_cap;
};

/**
 * @brief Represents a single Transformer layer.
 */
struct TransformerLayer {
  // Pointers to the tensor info for each weight in the layer
  TensorInfo* attn_norm_weight;
  TensorInfo* attn_q_weight;
  TensorInfo* attn_k_weight;
  TensorInfo* attn_v_weight;
  TensorInfo* attn_output_weight;
  TensorInfo* ffn_norm_weight;
  TensorInfo* ffn_gate_weight;
  TensorInfo* ffn_up_weight;
  TensorInfo* ffn_down_weight;
  TensorInfo* post_attention_norm_weight;
  TensorInfo* post_ffw_norm_weight;
  TensorInfo* attn_k_norm_weight;
  TensorInfo* attn_q_norm_weight;
};

/**
 * @brief The main Model class responsible for inference.
 */
class Model {
 public:
  /**
   * @brief Constructs a Model from a GGUF file.
   * @param gguf_file The GGUF file containing model weights and metadata.
   */
  Model(GGUFFile& gguf_file);

  const ModelHParams& hparams() const { return hparams_; }
  const std::vector<TransformerLayer>& layers() const { return layers_; }
  const TensorInfo* token_embd_weight() const { return token_embd_weight_; }
  const TensorInfo* output_norm_weight() const { return output_norm_weight_; }

  /**
   * @brief Performs a forward pass on the model.
   * @param tokens Input tokens.
   * @param pos Current position in the sequence.
   * @return Logits for the input tokens.
   */
  std::vector<std::vector<float>> forward(const std::vector<int>& tokens,
                                          int pos);

  /**
   * @brief Embeds tokens into vectors.
   * @param tokens Input tokens.
   * @return Embedded vectors.
   */
  tensor_2 embed_tokens(const std::vector<int>& tokens);

  /**
   * @brief Scales embedding vectors.
   * @param embeddings Embedding vectors to scale in-place.
   */
  void scale_embeddings(tensor_2& embeddings);

  /**
   * @brief Tokenizes a prompt string.
   * @param prompt The input string.
   * @param apply_chat_template Whether to apply the chat template.
   * @return Vector of token IDs.
   */
  std::vector<int> tokenize(const std::string& prompt,
                            bool apply_chat_template);

 private:
  void load_hparams(GGUFFile& gguf_file);
  void map_tensors(GGUFFile& gguf_file);
  void load_vocabulary();
  tensor_1 run_norm(const tensor_1& input, const TensorInfo* norm_weight,
                      int layer_id);
  tensor_2 run_norm(const tensor_2& input, const TensorInfo* norm_weight,
                      int layer_id);
  tensor_3 run_norm(const tensor_3& input, const TensorInfo* norm_weight,
                      int layer_id);
  tensor_2 run_attn(KVCacheLayer& kv_cache, const TensorInfo* output_weights,
                    const tensor_3& q_heads, const tensor_3& k_heads,
                    const tensor_3& v_heads, uint32_t n_head,
                    uint32_t n_head_kv, uint32_t n_embd_head, int layer_index,
                    int pos);

  ModelHParams hparams_;
  std::vector<TransformerLayer> layers_;
  TensorInfo* token_embd_weight_;
  TensorInfo* output_norm_weight_;
  GGUFFile& gguf_file_;
  KVCache kv_cache_;
  tensor_1 token_embd_weight_f32_;

  std::map<std::string, int> token_to_id;
  std::vector<std::string> id_to_token;
  int bos_token_id = -1;
  int unk_token_id = -1;
  size_t max_token_len = 0;
};

#endif  // MODEL_H
