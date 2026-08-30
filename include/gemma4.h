#pragma once

#include "safetensors.h"
#include "simd_avx2.h"
#include <string>
#include <vector>
#include <memory>

namespace gemma4 {

enum class LayerType {
    SLIDING_ATTENTION,
    FULL_ATTENTION
};

struct Gemma4Config {
    int vocab_size = 262144;
    int hidden_size = 5376;
    int intermediate_size = 21504;
    int num_hidden_layers = 60;
    
    // Sliding Attention config
    int num_attention_heads = 32;
    int num_key_value_heads = 16;
    int head_dim = 256;
    int sliding_window = 1024;
    float rope_theta_sliding = 10000.0f;
    
    // Full Attention config
    int num_global_key_value_heads = 4;
    int global_head_dim = 512;
    float rope_theta_full = 1000000.0f;
    float partial_rotary_factor_full = 0.25f; // 128 of 512
    bool attention_k_eq_v = true;
    
    int max_position_embeddings = 262144;
    float rms_norm_eps = 1e-6f;
    float final_logit_softcapping = 30.0f;
    std::vector<LayerType> layer_types;
};

struct LayerWeights {
    const uint16_t* input_layernorm = nullptr;
    const uint16_t* post_attention_layernorm = nullptr;
    const uint16_t* pre_feedforward_layernorm = nullptr;
    const uint16_t* post_feedforward_layernorm = nullptr;
    const uint16_t* layer_scalar = nullptr;
    
    const uint16_t* q_proj = nullptr;
    const uint16_t* k_proj = nullptr;
    const uint16_t* v_proj = nullptr;
    const uint16_t* o_proj = nullptr;
    const uint16_t* q_norm = nullptr;
    const uint16_t* k_norm = nullptr;
    
    const uint16_t* gate_proj = nullptr;
    const uint16_t* up_proj = nullptr;
    const uint16_t* down_proj = nullptr;
    
    LayerType type = LayerType::SLIDING_ATTENTION;
};

struct KVCache {
    std::vector<float> k_cache; // [max_seq_len x kv_dim]
    std::vector<float> v_cache; // [max_seq_len x kv_dim]
    int cur_len = 0;
};

class Gemma4Model {
public:
    Gemma4Model();
    ~Gemma4Model();

    bool load(const std::string& model_dir);
    void forward(int token_id, int pos, float* out_logits);
    void forward_batch(const std::vector<int>& tokens, int start_pos, float* out_logits_last);
    void reset_cache();

    const Gemma4Config& get_config() const { return config; }

private:
    Gemma4Config config;
    SafetensorsLoader loader;
    
    const uint16_t* embed_tokens = nullptr;
    const uint16_t* final_norm = nullptr;
    
    std::vector<LayerWeights> layers;
    std::vector<KVCache> kv_caches;
    
    // Scratch activation buffers
    std::vector<float> x_buf;              // [5376]
    std::vector<float> x_norm;             // [5376]
    std::vector<float> q_buf;              // [16384]
    std::vector<float> k_buf;              // [4096]
    std::vector<float> v_buf;              // [4096]
    std::vector<float> attn_out;           // [5376]
    std::vector<float> head_outputs;       // [16384]
    std::vector<float> gate_buf;           // [21504]
    std::vector<float> up_buf;             // [21504]
    std::vector<float> mlp_out;            // [5376]
    std::vector<float> thread_scores;      // [32 x 1024]

    // Batched scratch buffers
    std::vector<float> batch_x;            // [batch_size x 5376]
    std::vector<float> batch_norm;         // [batch_size x 5376]
    std::vector<float> batch_q;            // [batch_size x 16384]
    std::vector<float> batch_k;            // [batch_size x 4096]
    std::vector<float> batch_v;            // [batch_size x 4096]
    std::vector<float> batch_gate;         // [batch_size x 21504]
    std::vector<float> batch_up;           // [batch_size x 21504]
    std::vector<float> batch_mlp_out;      // [batch_size x 5376]
    std::vector<float> batch_attn_out;     // [batch_size x 5376]
    std::vector<float> batch_head_outs;    // [batch_size x 16384]
};

} // namespace gemma4
