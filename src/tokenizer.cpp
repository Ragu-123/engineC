#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

namespace gemma4 {

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Warning: Could not open " << tokenizer_json_path << ", using fallback character tokenizer" << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Parse vocab mapping from "vocab": { ... } in tokenizer.json
    size_t vocab_pos = content.find("\"vocab\":");
    if (vocab_pos != std::string::npos) {
        size_t open_brace = content.find("{", vocab_pos);
        size_t close_brace = content.find("}", open_brace);
        if (open_brace != std::string::npos && close_brace != std::string::npos) {
            std::string vocab_str = content.substr(open_brace + 1, close_brace - open_brace - 1);
            size_t p = 0;
            while (p < vocab_str.size()) {
                size_t q1 = vocab_str.find("\"", p);
                if (q1 == std::string::npos) break;
                size_t q2 = vocab_str.find("\"", q1 + 1);
                if (q2 == std::string::npos) break;
                std::string token = vocab_str.substr(q1 + 1, q2 - q1 - 1);
                
                size_t colon = vocab_str.find(":", q2 + 1);
                if (colon == std::string::npos) break;
                size_t comma = vocab_str.find_first_of(",}", colon + 1);
                if (comma == std::string::npos) comma = vocab_str.size();
                
                std::string id_str = vocab_str.substr(colon + 1, comma - colon - 1);
                try {
                    int id = std::stoi(id_str);
                    token_to_id[token] = id;
                    if ((size_t)id >= id_to_token.size()) {
                        id_to_token.resize(id + 1);
                    }
                    id_to_token[id] = token;
                } catch (...) {}
                p = comma + 1;
            }
        }
    }
    
    if (id_to_token.empty()) {
        // Fallback default vocab initialization
        id_to_token.resize(262144, "<unk>");
        id_to_token[0] = "<pad>";
        id_to_token[1] = "<eos>";
        id_to_token[2] = "<bos>";
    }
    
    std::cout << "[Tokenizer] Loaded vocabulary of size: " << id_to_token.size() << std::endl;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;
    if (add_bos) {
        tokens.push_back(bos_token_id);
    }
    
    // Greedy longest matching / BPE lookup
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        // Try prefixes up to length 32
        for (size_t len = std::min((size_t)32, text.size() - i); len >= 1; --len) {
            std::string sub = text.substr(i, len);
            auto it = token_to_id.find(sub);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Encode as byte token or fallback
            tokens.push_back((unsigned char)text[i]);
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
    
    // Temperature scaling
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
    
    // Top-P filtering
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
