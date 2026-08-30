# 🚀 engineC: Zero-Copy Gemma-4 31B Custom C++ CPU Inference Engine

A high-performance, standalone C++ inference engine for **Google Gemma-4 31B** (32.68B parameters) designed to run CPU inference with zero quantization using **Zero-Copy Memory-Mapping (`mmap`)**, **AVX2 + FMA SIMD Vectorization**, and **OpenMP Multithreading**.

---

## 🌟 Key Features

* **Zero Quantization / Pure BF16 Execution**: Directly reads bfloat16 Safetensors without quantizing or modifying model weights.
* **Zero-Copy 64-bit Memory-Mapping (`mmap`)**: Maps 58.3 GB of Safetensors weights into virtual address space with zero upfront physical RAM allocation. Clean pages are dynamically streamed and evicted by the Linux OS kernel, fitting comfortably within 31 GB RAM.
* **AVX2 + FMA SIMD Vector Kernels**:
  * In-register bfloat16-to-float32 unpacking (`_mm256_slli_epi32`).
  * 8-wide Fused Multiply-Add (`_mm256_fmadd_ps`) matrix-vector multiplication ($\text{GEMV}$).
  * Vectorized Gemma RMSNorm with $(1.0 + w)$ scaling factor.
  * Fast polynomial approximation for `gelu_pytorch_tanh` GeGLU activation.
* **Gemma-4 Exact Architecture**:
  * 60 Transformer layers (10 blocks of 5 Sliding Attention [window 1024] + 1 Full Attention [Proportional RoPE]).
  * GQA: 32 Query heads, 16 Key/Value heads, `head_dim = 256`.
  * Per-head QK-RMSNorm (`q_norm`, `k_norm`).
  * Attention and Final Logit Softcapping ($30.0 \times \tanh(x / 30.0)$).
  * 4 LayerNorms per block (`input`, `post_attention`, `pre_feedforward`, `post_feedforward`) with `layer_scalar` scaling.
* **Static & Rolling KV-Cache**: Zero memory allocations during autoregressive token generation.

---

## 🛠️ Project Structure

```text
engineC/
├── Makefile                      # Optimized build flags (-O3, -mavx2, -mfma, -fopenmp, -march=native)
├── README.md                     # Documentation & usage guide
├── .gitignore
├── include/
│   ├── gemma4.h                  # Model hyperparameters, layer types & forward pass interface
│   ├── safetensors.h             # Multi-shard zero-copy mmap parser & tensor address resolver
│   ├── simd_avx2.h               # AVX2/FMA BF16->FP32 GEMV, RMSNorm, GELU, RoPE intrinsics
│   └── tokenizer.h               # Fast BPE tokenizer and greedy/temperature sampler
└── src/
    ├── main.cpp                  # CLI entry point, benchmark timers, and generation loop
    ├── gemma4.cpp                # 60-layer forward pass & sliding/full attention pipeline
    ├── safetensors.cpp           # Safetensors mmap loader & page prefetching
    └── tokenizer.cpp             # BPE vocabulary reader & token encoder/decoder
```

---

## ⚡ Quick Build & Execution

### 1. Clone & Build
```bash
git clone https://github.com/Ragu-123/engineC.git
cd engineC
make -j4
```

### 2. Run Inference
```bash
./gemma4_cpu_engine \
  --model /kaggle/input/models/google/gemma-4/transformers/gemma-4-31b/1 \
  --prompt "Explain how solar filaments are formed on the surface of the Sun in detail:" \
  --tokens 64 \
  --temp 0.7
```

---

## 📊 Hardware Profile & Requirements
* **CPU**: x86-64 with AVX2 & FMA (Intel Haswell+, AMD Zen+)
* **RAM**: 32 GB RAM (64-bit OS with virtual memory mmap support)
* **Compiler**: GCC 11+ / Clang 13+ with OpenMP support
