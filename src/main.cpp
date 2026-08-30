#include "gemma4.h"
#include "tokenizer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

void print_banner() {
    std::cout << "\n=======================================================\n";
    std::cout << "  🚀 engineC: Zero-Copy Gemma-4 31B CPU Inference Engine\n";
    std::cout << "  ⚡ Hardware Vectorization: AVX2 + FMA + OpenMP\n";
    std::cout << "  📦 Pure Zero-Copy mmap | 32.68B Parameters\n";
    std::cout << "  ⚡ High-Speed Batched Prefill & Real-Time Token Streaming\n";
    std::cout << "=======================================================\n\n" << std::flush;
}

int main(int argc, char** argv) {
    print_banner();

    std::string model_dir = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b/1";
    std::string prompt = "The Sun is";
    int max_new_tokens = 5;
    float temperature = 0.0f;
    float top_p = 0.9f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        }
    }

    std::cout << "[Config] Model Directory: " << model_dir << "\n";
    std::cout << "[Config] Max New Tokens:  " << max_new_tokens << "\n";
    std::cout << "[Config] Temperature:     " << temperature << " (Top-P: " << top_p << ")\n\n" << std::flush;

    // 1. Load Tokenizer
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    std::cout << "[Step 1/3] Loading Tokenizer from " << tokenizer_path << "...\n" << std::flush;
    gemma4::Tokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path)) {
        std::cerr << "Fatal Error: Failed to load tokenizer from " << tokenizer_path << std::endl;
        return 1;
    }

    // 2. Zero-Copy Map Gemma-4 31B
    std::cout << "\n[Step 2/3] Zero-Copy Memory-Mapping Gemma-4 31B Safetensors...\n" << std::flush;
    gemma4::Gemma4Model model;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!model.load(model_dir)) {
        std::cerr << "\nFatal Error: Failed to load model from " << model_dir << std::endl;
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_s = std::chrono::duration<double>(t_load_end - t_load_start).count();
    std::cout << "\n✅ Model mapped in " << std::fixed << std::setprecision(2) << load_time_s << " seconds!\n\n" << std::flush;

    // 3. Tokenize Prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true);
    std::cout << "[Step 3/3] Prompt (" << prompt_tokens.size() << " tokens): \"" << prompt << "\"\n\n" << std::flush;

    gemma4::Sampler sampler(temperature, top_p);
    std::vector<float> logits(model.get_config().vocab_size, 0.0f);
    model.reset_cache();

    // ⚡ Phase 1: Batched Parallel Prefill
    std::cout << "[Prefill] Processing " << prompt_tokens.size() << " prompt tokens with parallel GEMM...\n" << std::flush;
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
              << " s (" << prefill_tok_sec << " tokens/sec)!\n\n" << std::flush;

    // ⚡ Phase 2: Generation Loop with Real-Time Streaming
    std::cout << "--- Starting Generation Output ---\n" << std::flush;
    std::cout << prompt << std::flush;

    int cur_token = sampler.sample(logits.data(), logits.size());
    int cur_pos = static_cast<int>(prompt_tokens.size());
    int generated_count = 0;

    auto t_gen_start = std::chrono::high_resolution_clock::now();

    while (generated_count < max_new_tokens && cur_token != tokenizer.eos_id()) {
        std::string token_str = tokenizer.decode_token(cur_token);
        std::cout << token_str << std::flush;
        generated_count++;
        cur_pos++;

        if (generated_count >= max_new_tokens) break;

        model.forward(cur_token, cur_pos - 1, logits.data());
        cur_token = sampler.sample(logits.data(), logits.size());
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double total_gen_s = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    double gen_tok_sec = (total_gen_s > 0.0 && generated_count > 0) ? ((double)generated_count / total_gen_s) : 0.0;

    std::cout << "\n\n=======================================================\n";
    std::cout << "  📊 Performance Summary:\n";
    std::cout << "  ⚡ Prefill Speed:    " << std::fixed << std::setprecision(2) << prefill_tok_sec << " tokens/sec (" << prompt_tokens.size() << " tokens)\n";
    std::cout << "  ⚡ Generation Speed: " << std::fixed << std::setprecision(2) << gen_tok_sec << " tokens/sec (" << generated_count << " tokens)\n";
    std::cout << "  ⏱️ Total Latency:    " << std::fixed << std::setprecision(2) << total_gen_s << " s\n";
    std::cout << "=======================================================\n\n" << std::flush;

    return 0;
}
