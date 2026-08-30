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
    int num_attention_heads = 32;
    int num_key_value_heads = 16;
    int head_dim = 256;
    int global_head_dim = 512;
    int max_position_embeddings = 262144;
    int sliding_window = 1024;
    float rms_norm_eps = 1e-6f;
    float final_logit_softcapping = 30.0f;
    float rope_theta_sliding = 10000.0f;
    float rope_theta_full = 1000000.0f;
    float partial_rotary_factor_full = 0.25f; // 64 of 256
    std::vector<LayerType> layer_types;
};

// Layer weight pointers directly resolved into mmap address space
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

// Static & Rolling Key-Value Cache per Layer
struct KVCache {
    std::vector<float> k_cache; // [max_seq_len x (num_kv_heads * head_dim)]
    std::vector<float> v_cache; // [max_seq_len x (num_kv_heads * head_dim)]
    int cur_len = 0;
};

class Gemma4Model {
public:
    Gemma4Model();
    ~Gemma4Model();

    // Initialize config and resolve all tensor pointers from Safetensors mmap
    bool load(const std::string& model_dir);

    // Single-token forward pass (Decode step)
    void forward(int token_id, int pos, float* out_logits);

    // High-Throughput Batched Forward Pass (Prefill step / Multi-token verification)
    // Computes logits for all tokens in `tokens` simultaneously using parallel GEMM!
    void forward_batch(const std::vector<int>& tokens, int start_pos, float* out_logits_last);

    // Reset KV-cache for a fresh generation sequence
    void reset_cache();

    // Asynchronously prefetch layer weights ahead of compute
    void prefetch_layer(int layer_idx);

    const Gemma4Config& get_config() const { return config; }

private:
    Gemma4Config config;
    SafetensorsLoader loader;
    
    const uint16_t* embed_tokens = nullptr;
    const uint16_t* final_norm = nullptr;
    
    std::vector<LayerWeights> layers;
    std::vector<KVCache> kv_caches;
    
    // Scratch activation buffers
    std::vector<float> x_buf;              // [hidden_size]
    std::vector<float> x_norm;             // [hidden_size]
    std::vector<float> q_buf;              // [num_heads * head_dim] = [8192]
    std::vector<float> k_buf;              // [num_kv_heads * head_dim] = [4096]
    std::vector<float> v_buf;              // [num_kv_heads * head_dim] = [4096]
    std::vector<float> attn_out;           // [hidden_size] = [5376]
    std::vector<float> gate_buf;           // [intermediate_size] = [21504]
    std::vector<float> up_buf;             // [intermediate_size] = [21504]
    std::vector<float> mlp_out;            // [hidden_size] = [5376]
    std::vector<float> attn_scores;        // [max_context_len]

    // Batched scratch buffers for ultra-fast multi-token prefill
    std::vector<float> batch_x;            // [batch_size x hidden_size]
    std::vector<float> batch_norm;         // [batch_size x hidden_size]
    std::vector<float> batch_q;            // [batch_size x 8192]
    std::vector<float> batch_k;            // [batch_size x 4096]
    std::vector<float> batch_v;            // [batch_size x 4096]
    std::vector<float> batch_gate;         // [batch_size x 21504]
    std::vector<float> batch_up;           // [batch_size x 21504]
    std::vector<float> batch_mlp_out;      // [batch_size x 5376]
    std::vector<float> batch_attn_out;     // [batch_size x 5376]
};

} // namespace gemma4
