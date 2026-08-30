#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace gemma4 {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Load vocabulary from tokenizer.json
    bool load(const std::string& tokenizer_json_path);

    // Simple word/subword encoder
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs back to string
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode_token(int token_id) const;

    // Special token IDs
    int bos_id() const { return bos_token_id; }
    int eos_id() const { return eos_token_id; }
    int pad_id() const { return pad_token_id; }
    size_t vocab_size() const { return id_to_token.size(); }

private:
    int bos_token_id = 2;
    int eos_token_id = 1;
    int pad_token_id = 0;

    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
};

// Sampler: Greedy, Temperature, Top-P
class Sampler {
public:
    Sampler(float temperature = 0.7f, float top_p = 0.9f);
    
    // Sample a token ID from logits array
    int sample(float* logits, size_t vocab_size);

private:
    float temperature;
    float top_p;
};

} // namespace gemma4
