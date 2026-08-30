#include "gemma4.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace gemma4 {

Gemma4Model::Gemma4Model() {
    // 60 layers pattern: 10 repetitions of (5 sliding + 1 full attention)
    config.layer_types.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        if (i % 6 == 5) {
            config.layer_types[i] = LayerType::FULL_ATTENTION;
        } else {
            config.layer_types[i] = LayerType::SLIDING_ATTENTION;
        }
    }
}

Gemma4Model::~Gemma4Model() {}

bool Gemma4Model::load(const std::string& model_dir) {
    if (!loader.load_directory(model_dir)) {
        return false;
    }
    
    // Bind Embed Tokens (tied to LM Head)
    const auto* embed_info = loader.get_tensor("model.language_model.embed_tokens.weight");
    if (!embed_info) {
        std::cerr << "[Gemma4Model] Error: Missing embed_tokens tensor!" << std::endl;
        return false;
    }
    embed_tokens = embed_info->ptr_bf16;
    
    // Bind Final Norm
    const auto* norm_info = loader.get_tensor("model.language_model.norm.weight");
    if (!norm_info) {
        std::cerr << "[Gemma4Model] Error: Missing final norm tensor!" << std::endl;
        return false;
    }
    final_norm = norm_info->ptr_bf16;
    
    // Bind all 60 layers
    layers.resize(config.num_hidden_layers);
    kv_caches.resize(config.num_hidden_layers);
    
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        std::string pfx = "model.language_model.layers." + std::to_string(i) + ".";
        
        auto get_ptr = [&](const std::string& name) -> const uint16_t* {
            const auto* t = loader.get_tensor(pfx + name);
            return t ? t->ptr_bf16 : nullptr;
        };
        
        layers[i].input_layernorm = get_ptr("input_layernorm.weight");
        layers[i].post_attention_layernorm = get_ptr("post_attention_layernorm.weight");
        layers[i].pre_feedforward_layernorm = get_ptr("pre_feedforward_layernorm.weight");
        layers[i].post_feedforward_layernorm = get_ptr("post_feedforward_layernorm.weight");
        layers[i].layer_scalar = get_ptr("layer_scalar");
        
        layers[i].q_proj = get_ptr("self_attn.q_proj.weight");
        layers[i].k_proj = get_ptr("self_attn.k_proj.weight");
        layers[i].v_proj = get_ptr("self_attn.v_proj.weight");
        layers[i].o_proj = get_ptr("self_attn.o_proj.weight");
        layers[i].q_norm = get_ptr("self_attn.q_norm.weight");
        layers[i].k_norm = get_ptr("self_attn.k_norm.weight");
        
        layers[i].gate_proj = get_ptr("mlp.gate_proj.weight");
        layers[i].up_proj = get_ptr("mlp.up_proj.weight");
        layers[i].down_proj = get_ptr("mlp.down_proj.weight");
        
        layers[i].type = config.layer_types[i];
        
        // Allocate KV-cache: max 4096 context tokens for CPU generation
        size_t kv_dim = config.num_key_value_heads * config.head_dim; // 16 * 256 = 4096 floats
        kv_caches[i].k_cache.resize(4096 * kv_dim, 0.0f);
        kv_caches[i].v_cache.resize(4096 * kv_dim, 0.0f);
        kv_caches[i].cur_len = 0;
    }
    
    // Allocate activation scratchpads
    x_buf.resize(config.hidden_size);
    x_norm.resize(config.hidden_size);
    q_buf.resize(config.num_attention_heads * config.head_dim); // 32 * 256 = 8192
    k_buf.resize(config.num_key_value_heads * config.head_dim); // 16 * 256 = 4096
    v_buf.resize(config.num_key_value_heads * config.head_dim); // 16 * 256 = 4096
    attn_out.resize(config.hidden_size);
    gate_buf.resize(config.intermediate_size);                  // 21504
    up_buf.resize(config.intermediate_size);                    // 21504
    mlp_out.resize(config.hidden_size);
    attn_scores.resize(4096);
    
    std::cout << "[Gemma4Model] Successfully bound all 60 transformer layers with zero-copy mmap!" << std::endl;
    return true;
}

void Gemma4Model::reset_cache() {
    for (auto& cache : kv_caches) {
        cache.cur_len = 0;
    }
}

void Gemma4Model::prefetch_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= config.num_hidden_layers) return;
    std::string pfx = "model.language_model.layers." + std::to_string(layer_idx) + ".";
    loader.prefetch_tensor(pfx + "self_attn.q_proj.weight");
    loader.prefetch_tensor(pfx + "mlp.gate_proj.weight");
}

void Gemma4Model::forward(int token_id, int pos, float* out_logits) {
    int H = config.hidden_size;             // 5376
    int I = config.intermediate_size;       // 21504
    int n_q_heads = config.num_attention_heads;     // 32
    int n_kv_heads = config.num_key_value_heads;   // 16
    int head_dim = config.head_dim;         // 256
    int kv_dim = n_kv_heads * head_dim;     // 4096
    int q_dim = n_q_heads * head_dim;       // 8192
    int gqa_ratio = n_q_heads / n_kv_heads; // 2
    
    // 1. Embedding lookup & scaling by sqrt(d_model)
    const uint16_t* emb_row = embed_tokens + (size_t)token_id * H;
    float emb_scale = sqrtf((float)H);
    for (int i = 0; i < H; ++i) {
        uint32_t val = (uint32_t)emb_row[i] << 16;
        float fw;
        memcpy(&fw, &val, 4);
        x_buf[i] = fw * emb_scale;
    }
    
    // 2. Pass through 60 Transformer Layers
    for (int l = 0; l < config.num_hidden_layers; ++l) {
        const auto& layer = layers[l];
        
        // Prefetch next layer in background
        if (l + 1 < config.num_hidden_layers) {
            prefetch_layer(l + 1);
        }
        
        // Read layer scalar multiplier
        float scalar = 1.0f;
        if (layer.layer_scalar) {
            uint32_t val = (uint32_t)(*layer.layer_scalar) << 16;
            memcpy(&scalar, &val, 4);
        }
        
        // --- A. Self-Attention Block ---
        // Input RMSNorm
        simd::rmsnorm_gemma(x_buf.data(), layer.input_layernorm, x_norm.data(), H, config.rms_norm_eps);
        
        // Q, K, V Projections
        simd::gemv_bf16_f32(layer.q_proj, x_norm.data(), q_buf.data(), q_dim, H);
        simd::gemv_bf16_f32(layer.k_proj, x_norm.data(), k_buf.data(), kv_dim, H);
        simd::gemv_bf16_f32(layer.v_proj, x_norm.data(), v_buf.data(), kv_dim, H);
        
        // Per-Head QK-Norm
        for (int h = 0; h < n_q_heads; ++h) {
            simd::rmsnorm_gemma(q_buf.data() + h * head_dim, layer.q_norm, q_buf.data() + h * head_dim, head_dim, config.rms_norm_eps);
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            simd::rmsnorm_gemma(k_buf.data() + h * head_dim, layer.k_norm, k_buf.data() + h * head_dim, head_dim, config.rms_norm_eps);
        }
        
        // RoPE (Rotary Position Embeddings)
        if (layer.type == LayerType::FULL_ATTENTION) {
            // Full attention: Proportional RoPE with rotary_dim = 64 of 256
            size_t rotary_dim = (size_t)(config.partial_rotary_factor_full * head_dim);
            simd::apply_rope(q_buf.data(), k_buf.data(), n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, config.rope_theta_full);
        } else {
            // Sliding attention: standard RoPE with rotary_dim = head_dim
            simd::apply_rope(q_buf.data(), k_buf.data(), n_q_heads, n_kv_heads, head_dim, head_dim, pos, config.rope_theta_sliding);
        }
        
        // Update KV-Cache
        auto& cache = kv_caches[l];
        int store_idx = pos % 4096;
        memcpy(cache.k_cache.data() + store_idx * kv_dim, k_buf.data(), kv_dim * sizeof(float));
        memcpy(cache.v_cache.data() + store_idx * kv_dim, v_buf.data(), kv_dim * sizeof(float));
        if (pos >= cache.cur_len) cache.cur_len = pos + 1;
        
        // Multi-Head Attention Computation
        int window_size = (layer.type == LayerType::SLIDING_ATTENTION) ? config.sliding_window : 4096;
        int start_pos = std::max(0, pos - window_size + 1);
        int num_context = pos - start_pos + 1;
        float qk_scale = 1.0f / sqrtf((float)head_dim);
        
        std::vector<float> head_outputs(q_dim, 0.0f);
        
        #pragma omp parallel for schedule(static)
        for (int qh = 0; qh < n_q_heads; ++qh) {
            int kvh = qh / gqa_ratio;
            const float* q_ptr = q_buf.data() + qh * head_dim;
            
            // Compute attention scores against context history
            std::vector<float> local_scores(num_context);
            float max_score = -1e9f;
            
            for (int ci = 0; ci < num_context; ++ci) {
                int ctx_pos = start_pos + ci;
                const float* k_ptr = cache.k_cache.data() + (ctx_pos % 4096) * kv_dim + kvh * head_dim;
                
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                float s = dot * qk_scale;
                // Attention Softcapping (30.0 * tanh(s / 30.0))
                s = 30.0f * tanhf(s / 30.0f);
                local_scores[ci] = s;
                if (s > max_score) max_score = s;
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (int ci = 0; ci < num_context; ++ci) {
                float exp_val = expf(local_scores[ci] - max_score);
                local_scores[ci] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            
            // Weighted sum over V values
            float* out_h = head_outputs.data() + qh * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] = 0.0f;
            }
            for (int ci = 0; ci < num_context; ++ci) {
                int ctx_pos = start_pos + ci;
                float weight = local_scores[ci] * inv_sum;
                const float* v_ptr = cache.v_cache.data() + (ctx_pos % 4096) * kv_dim + kvh * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_h[d] += weight * v_ptr[d];
                }
            }
        }
        
        // Output Projection: attn_out = W_o * head_outputs
        simd::gemv_bf16_f32(layer.o_proj, head_outputs.data(), attn_out.data(), H, q_dim);
        
        // Post-Attention LayerNorm & Residual
        simd::rmsnorm_gemma(attn_out.data(), layer.post_attention_layernorm, attn_out.data(), H, config.rms_norm_eps);
        for (int i = 0; i < H; ++i) {
            x_buf[i] += attn_out[i] * scalar;
        }
        
        // --- B. Feedforward (MLP) Block ---
        // Pre-Feedforward LayerNorm
        simd::rmsnorm_gemma(x_buf.data(), layer.pre_feedforward_layernorm, x_norm.data(), H, config.rms_norm_eps);
        
        // Gate & Up Projections
        simd::gemv_bf16_f32(layer.gate_proj, x_norm.data(), gate_buf.data(), I, H);
        simd::gemv_bf16_f32(layer.up_proj, x_norm.data(), up_buf.data(), I, H);
        
        // GeGLU Activation: gate = gelu(gate) * up
        simd::geglu_activation(gate_buf.data(), up_buf.data(), I);
        
        // Down Projection: mlp_out = W_down * gate
        simd::gemv_bf16_f32(layer.down_proj, gate_buf.data(), mlp_out.data(), H, I);
        
        // Post-Feedforward LayerNorm & Residual
        simd::rmsnorm_gemma(mlp_out.data(), layer.post_feedforward_layernorm, mlp_out.data(), H, config.rms_norm_eps);
        for (int i = 0; i < H; ++i) {
            x_buf[i] += mlp_out[i] * scalar;
        }
    }
    
    // 3. Final LayerNorm
    simd::rmsnorm_gemma(x_buf.data(), final_norm, x_norm.data(), H, config.rms_norm_eps);
    
    // 4. Output Logits (Tied LM Head)
    simd::gemv_bf16_f32(embed_tokens, x_norm.data(), out_logits, config.vocab_size, H);
    
    // 5. Final Logit Softcapping (30.0 * tanh(x / 30.0))
    if (config.final_logit_softcapping > 0.0f) {
        simd::softcap_logits(out_logits, config.vocab_size, config.final_logit_softcapping);
    }
}

} // namespace gemma4
