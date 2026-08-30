#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>

namespace gemma4 {
namespace simd {

// Fast approximate tanh using polynomial / rational approximation for AVX2
static inline __m256 fast_tanh_ps(__m256 x) {
    // Clamp x to [-9.0, 9.0] to prevent overflow
    __m256 max_val = _mm256_set1_ps(9.0f);
    __m256 min_val = _mm256_set1_ps(-9.0f);
    x = _mm256_max_ps(min_val, _mm256_min_ps(max_val, x));
    
    __m256 x2 = _mm256_mul_ps(x, x);
    // Padé (3,3) approximant for tanh
    __m256 num = _mm256_fmadd_ps(x2, _mm256_set1_ps(0.1333333333f), _mm256_set1_ps(1.0f));
    num = _mm256_mul_ps(x, num);
    __m256 den = _mm256_fmadd_ps(x2, _mm256_set1_ps(0.4666666667f), _mm256_set1_ps(1.0f));
    return _mm256_div_ps(num, den);
}

// Vectorized AVX2 + FMA dot-product of BFloat16 weights with Float32 vector
static inline float dot_bf16_f32(const uint16_t* __restrict__ w_bf16, const float* __restrict__ x_f32, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128i raw0 = _mm_loadu_si128((const __m128i*)(w_bf16 + i));
        __m128i raw1 = _mm_loadu_si128((const __m128i*)(w_bf16 + i + 8));
        
        __m256i int0 = _mm256_cvtepu16_epi32(raw0);
        __m256i int1 = _mm256_cvtepu16_epi32(raw1);
        
        __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(int0, 16));
        __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(int1, 16));
        
        __m256 x0 = _mm256_loadu_ps(x_f32 + i);
        __m256 x1 = _mm256_loadu_ps(x_f32 + i + 8);
        
        acc0 = _mm256_fmadd_ps(w0, x0, acc0);
        acc1 = _mm256_fmadd_ps(w1, x1, acc1);
    }
    
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float total = _mm_cvtss_f32(sum1);
    
    for (; i < n; ++i) {
        uint32_t val = (uint32_t)w_bf16[i] << 16;
        float fw;
        memcpy(&fw, &val, 4);
        total += fw * x_f32[i];
    }
    return total;
}

// Parallel Matrix-Vector Multiplication: y = W * x (W: [out_dim x in_dim] in BF16, x: [in_dim] in FP32)
static inline void gemv_bf16_f32(const uint16_t* __restrict__ W_bf16,
                                 const float* __restrict__ x_f32,
                                 float* __restrict__ y_f32,
                                 size_t out_dim,
                                 size_t in_dim) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < out_dim; ++r) {
        y_f32[r] = dot_bf16_f32(W_bf16 + r * in_dim, x_f32, in_dim);
    }
}

// Vectorized Gemma RMSNorm with (1.0 + weight) scaling factor: y = (x / sqrt(mean(x^2) + eps)) * (1.0 + w)
static inline void rmsnorm_gemma(const float* __restrict__ x,
                                 const uint16_t* __restrict__ w_bf16,
                                 float* __restrict__ out,
                                 size_t dim,
                                 float eps = 1e-6f) {
    __m256 sum_sq = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(vx, vx, sum_sq);
    }
    
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s4);
    __m128 s2 = _mm_add_ps(s4, sh);
    sh = _mm_movehl_ps(sh, s2);
    __m128 s1 = _mm_add_ss(s2, sh);
    float total_sq = _mm_cvtss_f32(s1);
    
    for (; i < dim; ++i) {
        total_sq += x[i] * x[i];
    }
    
    float inv_rms = 1.0f / sqrtf((total_sq / (float)dim) + eps);
    __m256 vinv = _mm256_set1_ps(inv_rms);
    __m256 one = _mm256_set1_ps(1.0f);
    
    i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m128i raw_w = _mm_loadu_si128((const __m128i*)(w_bf16 + i));
        __m256i int_w = _mm256_cvtepu16_epi32(raw_w);
        __m256 vw = _mm256_castsi256_ps(_mm256_slli_epi32(int_w, 16));
        
        __m256 scale = _mm256_add_ps(one, vw);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), scale);
        _mm256_storeu_ps(out + i, res);
    }
    
    for (; i < dim; ++i) {
        uint32_t val = (uint32_t)w_bf16[i] << 16;
        float fw;
        memcpy(&fw, &val, 4);
        out[i] = (x[i] * inv_rms) * (1.0f + fw);
    }
}

// Vectorized GeGLU Activation: gate = gelu_pytorch_tanh(gate) * up
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static inline void geglu_activation(float* __restrict__ gate, const float* __restrict__ up, size_t dim) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_sqrt = _mm256_set1_ps(sqrt_2_over_pi);
    __m256 v_coeff = _mm256_set1_ps(coeff);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        
        __m256 g2 = _mm256_mul_ps(g, g);
        __m256 g3 = _mm256_mul_ps(g2, g);
        __m256 inner = _mm256_fmadd_ps(v_coeff, g3, g);
        inner = _mm256_mul_ps(v_sqrt, inner);
        
        __m256 tanh_val = fast_tanh_ps(inner);
        __m256 gelu_g = _mm256_mul_ps(v_half, _mm256_mul_ps(g, _mm256_add_ps(v_one, tanh_val)));
        
        __m256 res = _mm256_mul_ps(gelu_g, u);
        _mm256_storeu_ps(gate + i, res);
    }
}

// Logit Softcapping: x = cap * tanh(x / cap)
static inline void softcap_logits(float* __restrict__ logits, size_t dim, float cap = 30.0f) {
    __m256 v_cap = _mm256_set1_ps(cap);
    __m256 v_inv_cap = _mm256_set1_ps(1.0f / cap);
    
    for (size_t i = 0; i + 7 < dim; i += 8) {
        __m256 x = _mm256_loadu_ps(logits + i);
        __m256 scaled = _mm256_mul_ps(x, v_inv_cap);
        __m256 t = fast_tanh_ps(scaled);
        __m256 res = _mm256_mul_ps(v_cap, t);
        _mm256_storeu_ps(logits + i, res);
    }
}

// Rotary Position Embedding (RoPE) for partial rotary factor (e.g. rotary_dim = 64 of head_dim = 256)
static inline void apply_rope(float* __restrict__ q,
                              float* __restrict__ k,
                              size_t n_heads_q,
                              size_t n_heads_k,
                              size_t head_dim,
                              size_t rotary_dim,
                              int pos,
                              float rope_theta) {
    for (size_t h = 0; h < n_heads_q; ++h) {
        float* q_head = q + h * head_dim;
        for (size_t i = 0; i < rotary_dim; i += 2) {
            float freq = 1.0f / powf(rope_theta, (float)i / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            
            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i]     = q0 * cos_a - q1 * sin_a;
            q_head[i + 1] = q0 * sin_a + q1 * cos_a;
        }
    }
    
    for (size_t h = 0; h < n_heads_k; ++h) {
        float* k_head = k + h * head_dim;
        for (size_t i = 0; i < rotary_dim; i += 2) {
            float freq = 1.0f / powf(rope_theta, (float)i / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            
            float k0 = k_head[i];
            float k1 = k_head[i + 1];
            k_head[i]     = k0 * cos_a - k1 * sin_a;
            k_head[i + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

} // namespace simd
} // namespace gemma4
