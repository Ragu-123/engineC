#include "gemma4.h"
#include "tokenizer.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

void print_banner() {
    std::cout << "\n=======================================================\n";
    std::cout << "  🚀 engineC: Zero-Copy Gemma-4 31B CPU Inference Engine\n";
    std::cout << "  ⚡ Hardware Vectorization: AVX2 + FMA + OpenMP\n";
    std::cout << "  📦 Pure Zero-Copy mmap | 32.68B Parameters\n";
    std::cout << "  ⚡ High-Speed Batched Prefill & Real-Time Token Streaming\n";
    std::cout << "=======================================================\n\n";
}

int main(int argc, char** argv) {
    print_banner();

    std::string model_dir = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b/1";
    std::string prompt = "Explain the three interior layers of the Sun:";
    int max_new_tokens = 16;
    float temperature = 0.7f;
    float top_p = 0.9f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_new_tokens = std::stoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) top_p = std::stof(argv[++i]);
    }

    std::cout << "[Config] Model Directory: " << model_dir << "\n";
    std::cout << "[Config] Max New Tokens:  " << max_new_tokens << "\n";
    std::cout << "[Config] Temperature:     " << temperature << " (Top-P: " << top_p << ")\n\n";

    // 1. Load Tokenizer
    gemma4::Tokenizer tokenizer;
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    std::cout << "[Step 1/3] Loading Tokenizer from " << tokenizer_path << "..." << std::flush;
    tokenizer.load(tokenizer_path);

    // 2. Load Model via Zero-Copy mmap
    gemma4::Gemma4Model model;
    std::cout << "\n[Step 2/3] Zero-Copy Memory-Mapping Gemma-4 31B Safetensors..." << std::flush;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!model.load(model_dir)) {
        std::cerr << "\nFatal Error: Failed to load model from " << model_dir << std::endl;
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_s = std::chrono::duration<double>(t_load_end - t_load_start).count();
    std::cout << "\n✅ Model mapped in " << std::fixed << std::setprecision(2) << load_time_s << " seconds!\n\n";

    // 3. Tokenize Prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true);
    std::cout << "[Step 3/3] Prompt (" << prompt_tokens.size() << " tokens): \"" << prompt << "\"\n\n";

    gemma4::Sampler sampler(temperature, top_p);
    std::vector<float> logits(model.get_config().vocab_size, 0.0f);
    model.reset_cache();

    // ⚡ Phase 1: Batched Parallel Prefill
    std::cout << "[Prefill] Processing " << prompt_tokens.size() << " prompt tokens with parallel GEMM..." << std::flush;
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    
    if (prompt_tokens.size() > 1) {
        model.forward_batch(prompt_tokens, 0, logits.data());
    } else if (prompt_tokens.size() == 1) {
        model.forward(prompt_tokens[0], 0, logits.data());
    }
    
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    double prefill_tok_sec = (double)prompt_tokens.size() / (prefill_ms / 1000.0);
    std::cout << " Done in " << std::fixed << std::setprecision(2) << (prefill_ms / 1000.0) 
              << " s (" << prefill_tok_sec << " tokens/sec)!\n\n";

    // ⚡ Phase 2: Generation Loop with Real-Time Streaming
    std::cout << "--- Starting Generation Output ---\n";
    std::cout << prompt << std::flush;

    int cur_token = sampler.sample(logits.data(), logits.size());
    int cur_pos = static_cast<int>(prompt_tokens.size());
    int generated_count = 0;

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    while (generated_count < max_new_tokens && cur_token != tokenizer.eos_id()) {
        std::string token_str = tokenizer.decode_token(cur_token);
        std::cout << token_str << std::flush;

        generated_count++;
        cur_pos++;

        // Compute forward pass for next token
        auto t_tok_start = std::chrono::high_resolution_clock::now();
        model.forward(cur_token, cur_pos - 1, logits.data());
        cur_token = sampler.sample(logits.data(), logits.size());
        auto t_tok_end = std::chrono::high_resolution_clock::now();
        (void)t_tok_start; (void)t_tok_end;
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_time_s = std::chrono::duration<double>(t_decode_end - t_decode_start).count();

    std::cout << "\n\n=======================================================\n";
    std::cout << "  📊 GENERATION BENCHMARK SUMMARY\n";
    std::cout << "=======================================================\n";
    std::cout << "  ⚡ Batched Prefill Throughput:  " << std::fixed << std::setprecision(1) 
              << prefill_tok_sec << " tokens/sec (" << prefill_ms << " ms total)\n";
    if (generated_count > 0) {
        std::cout << "  🚀 Generation Speed:           " << std::fixed << std::setprecision(2)
                  << (generated_count / decode_time_s) << " tokens/sec\n";
        std::cout << "  ⏱️  Latency per Token:          " << std::fixed << std::setprecision(2)
                  << (decode_time_s * 1000.0 / generated_count) << " ms/token\n";
    }
    std::cout << "  📦 Total Generated Tokens:     " << generated_count << " tokens\n";
    std::cout << "=======================================================\n";

    return 0;
}
