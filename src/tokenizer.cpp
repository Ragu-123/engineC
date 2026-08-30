#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

namespace gemma4 {

Tokenizer::Tokenizer() {
    id_to_token.resize(262144, "<unk>");
    id_to_token[0] = "<pad>";
    id_to_token[1] = "<eos>";
    id_to_token[2] = "<bos>";
}

Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Warning: Could not open " << tokenizer_json_path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Locate "vocab": { in the "model" block
    size_t vocab_pos = content.find("\"vocab\":");
    if (vocab_pos == std::string::npos) {
        std::cerr << "[Tokenizer] Error: \"vocab\" block not found in tokenizer.json" << std::endl;
        return false;
    }
    
    size_t open_brace = content.find("{", vocab_pos);
    if (open_brace == std::string::npos) return false;
    
    size_t p = open_brace + 1;
    size_t content_size = content.size();
    size_t parsed_count = 0;
    
    while (p < content_size) {
        // Find token string start
        size_t q1 = content.find("\"", p);
        if (q1 == std::string::npos) break;
        
        // Handle escaped quotes within token
        size_t q2 = q1 + 1;
        while (q2 < content_size) {
            if (content[q2] == '\"' && content[q2 - 1] != '\\') {
                break;
            }
            q2++;
        }
        if (q2 >= content_size) break;
        
        std::string token = content.substr(q1 + 1, q2 - q1 - 1);
        
        // Find colon
        size_t colon = content.find(":", q2 + 1);
        if (colon == std::string::npos) break;
        
        // Find integer ID
        size_t id_start = content.find_first_of("0123456789", colon + 1);
        if (id_start == std::string::npos) break;
        
        size_t id_end = content.find_first_not_of("0123456789", id_start);
        if (id_end == std::string::npos) id_end = content_size;
        
        std::string id_str = content.substr(id_start, id_end - id_start);
        try {
            int id = std::stoi(id_str);
            token_to_id[token] = id;
            if ((size_t)id >= id_to_token.size()) {
                id_to_token.resize(id + 1, "<unk>");
            }
            id_to_token[id] = token;
            parsed_count++;
        } catch (...) {}
        
        // Check if vocab block closed
        size_t next_comma = content.find(",", id_end);
        size_t next_close = content.find("}", id_end);
        
        if (next_close < next_comma || next_comma == std::string::npos) {
            // Reached end of vocab dictionary
            break;
        }
        p = next_comma + 1;
    }
    
    std::cout << "[Tokenizer] Successfully loaded " << parsed_count << " vocabulary entries from tokenizer.json!" << std::endl;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;
    if (add_bos) {
        tokens.push_back(bos_token_id);
    }
    
    // Replace space with SentencePiece prefix token " " (\xe2\x96\x81)
    std::string sp_text = "";
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ') {
            sp_text += "\xe2\x96\x81";
        } else {
            sp_text += text[i];
        }
    }
    
    // Greedy longest matching
    size_t i = 0;
    while (i < sp_text.size()) {
        bool matched = false;
        for (size_t len = std::min((size_t)32, sp_text.size() - i); len >= 1; --len) {
            std::string sub = sp_text.substr(i, len);
            auto it = token_to_id.find(sub);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            tokens.push_back((unsigned char)sp_text[i]);
            i++;
        }
    }
    return tokens;
}

std::string Tokenizer::decode_token(int token_id) const {
    if (token_id >= 0 && (size_t)token_id < id_to_token.size()) {
        std::string s = id_to_token[token_id];
        // Replace SPIECE_UNDERLINE (" ") with standard space " "
        size_t p = 0;
        while ((p = s.find("\xe2\x96\x81", p)) != std::string::npos) {
            s.replace(p, 3, " ");
            p += 1;
        }
        return s;
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result = "";
    for (int t : tokens) {
        if (t == eos_token_id) break;
        if (t == bos_token_id || t == pad_token_id) continue;
        result += decode_token(t);
    }
    return result;
}

Sampler::Sampler(float temperature, float top_p)
    : temperature(temperature), top_p(top_p) {}

int Sampler::sample(float* logits, size_t vocab_size) {
    if (temperature <= 0.0f) {
        // Greedy decoding
        int best_id = 0;
        float best_logit = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_id = static_cast<int>(i);
            }
        }
        return best_id;
    }
    
    float inv_temp = 1.0f / temperature;
    float max_l = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    
    std::vector<std::pair<float, int>> probs;
    probs.reserve(vocab_size);
    double sum = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        float p = expf((logits[i] - max_l) * inv_temp);
        probs.push_back({p, static_cast<int>(i)});
        sum += p;
    }
    
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    double cum_sum = 0.0;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cum_sum += probs[i].first / sum;
        if (cum_sum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    
    static std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, cum_sum);
    double r = dist(rng);
    double running = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        running += probs[i].first / sum;
        if (running >= r) {
            return probs[i].second;
        }
    }
    return probs[0].second;
}

} // namespace gemma4
