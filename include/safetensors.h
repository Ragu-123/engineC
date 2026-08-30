#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace gemma4 {

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_offsets[2]; // start, end within file data buffer
    int file_idx;           // index of mmap file shard
    const uint16_t* ptr_bf16; // direct pointer to memory
};

class SafetensorsLoader {
public:
    SafetensorsLoader();
    ~SafetensorsLoader();

    // Load model from directory containing model.safetensors.index.json or .safetensors files
    bool load_directory(const std::string& dir_path);

    // Get tensor by name
    const TensorInfo* get_tensor(const std::string& name) const;
    
    // Check if tensor exists
    bool has_tensor(const std::string& name) const;

    // Asynchronously prefetch tensor data into OS page cache
    void prefetch_tensor(const std::string& name);

    // Evict or hint OS that tensor is not immediately needed
    void evict_tensor(const std::string& name);

    // Total mapped bytes
    size_t get_total_mapped_bytes() const { return total_mapped_bytes; }

private:
    struct MappedFile {
        std::string path;
        int fd;
        void* addr;
        size_t size;
        size_t header_len;
        const char* data_start;
    };

    std::vector<MappedFile> mapped_files;
    std::unordered_map<std::string, TensorInfo> tensors;
    size_t total_mapped_bytes;

    bool parse_json_header(int file_idx, const char* json_str, size_t len);
};

} // namespace gemma4
