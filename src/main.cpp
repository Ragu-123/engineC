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
    std::cout << "=======================================================\n\n";
}

int main(int argc, char** argv) {
    print_banner();

    std::string model_dir = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b/1";
    std::string prompt = "Explain how solar filaments are formed on the surface of the Sun in detail:";
    int max_new_tokens = 64;
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
    std::cout << "[Step 1/3] Loading Tokenizer from " << tokenizer_path << "..." << std::endl;
    tokenizer.load(tokenizer_path);

    // 2. Load Model via Zero-Copy mmap
    gemma4::Gemma4Model model;
    std::cout << "\n[Step 2/3] Zero-Copy Memory-Mapping Gemma-4 31B Safetensors..." << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!model.load(model_dir)) {
        std::cerr << "Fatal Error: Failed to load model from " << model_dir << std::endl;
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_s = std::chrono::duration<double>(t_load_end - t_load_start).count();
    std::cout << "Model mapped in " << std::fixed << std::setprecision(2) << load_time_s << " seconds!\n\n";

    // 3. Tokenize Prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true);
    std::cout << "[Step 3/3] Prompt (" << prompt_tokens.size() << " tokens): \"" << prompt << "\"\n\n";
    std::cout << "--- Starting Generation Output ---\n";
    std::cout << prompt << std::flush;

    gemma4::Sampler sampler(temperature, top_p);
    std::vector<float> logits(model.get_config().vocab_size);
    model.reset_cache();

    // Prefill Phase
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        model.forward(prompt_tokens[i], static_cast<int>(i), logits.data());
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_gen_start).count();

    // Autoregressive Decode Phase
    int cur_token = sampler.sample(logits.data(), logits.size());
    int cur_pos = static_cast<int>(prompt_tokens.size());
    int generated_count = 0;

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    while (generated_count < max_new_tokens && cur_token != tokenizer.eos_id()) {
        std::string token_str = tokenizer.decode_token(cur_token);
        std::cout << token_str << std::flush;

        model.forward(cur_token, cur_pos, logits.data());
        cur_token = sampler.sample(logits.data(), logits.size());
        cur_pos++;
        generated_count++;
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_time_s = std::chrono::duration<double>(t_decode_end - t_decode_start).count();

    std::cout << "\n\n--- Generation Completed ---\n";
    std::cout << "Prefill Latency:   " << prefill_ms << " ms (" 
              << (prompt_tokens.size() / (prefill_ms / 1000.0)) << " tokens/sec)\n";
    if (generated_count > 0) {
        std::cout << "Decode Throughput: " << (generated_count / decode_time_s) << " tokens/sec ("
                  << (decode_time_s * 1000.0 / generated_count) << " ms/token)\n";
    }
    std::cout << "Total Generated:   " << generated_count << " tokens\n";
    std::cout << "=======================================================\n";

    return 0;
}
