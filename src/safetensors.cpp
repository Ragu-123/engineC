#include "safetensors.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

namespace gemma4 {

SafetensorsLoader::SafetensorsLoader() : total_mapped_bytes(0) {}

SafetensorsLoader::~SafetensorsLoader() {
    for (auto& mf : mapped_files) {
        if (mf.addr && mf.addr != MAP_FAILED) {
            munmap(mf.addr, mf.size);
        }
        if (mf.fd >= 0) {
            close(mf.fd);
        }
    }
}

// Minimal fast JSON tokenizer / extractor for Safetensors metadata
bool SafetensorsLoader::parse_json_header(int file_idx, const char* json_str, size_t len) {
    std::string s(json_str, len);
    
    // Parse each tensor entry: "tensor_name": { "dtype": "...", "shape": [...], "data_offsets": [...] }
    size_t pos = 0;
    while (pos < s.length()) {
        size_t name_start = s.find("\"", pos);
        if (name_start == std::string::npos) break;
        size_t name_end = s.find("\"", name_start + 1);
        if (name_end == std::string::npos) break;
        
        std::string tensor_name = s.substr(name_start + 1, name_end - name_start - 1);
        pos = name_end + 1;
        
        if (tensor_name == "__metadata__") {
            // Skip __metadata__ block
            size_t close_brace = s.find("}", pos);
            if (close_brace != std::string::npos) pos = close_brace + 1;
            continue;
        }
        
        size_t block_start = s.find("{", pos);
        if (block_start == std::string::npos) break;
        size_t block_end = s.find("}", block_start + 1);
        if (block_end == std::string::npos) break;
        
        std::string block = s.substr(block_start, block_end - block_start + 1);
        pos = block_end + 1;
        
        TensorInfo info;
        info.name = tensor_name;
        info.file_idx = file_idx;
        
        // Extract dtype
        size_t dt_pos = block.find("\"dtype\":");
        if (dt_pos != std::string::npos) {
            size_t q1 = block.find("\"", dt_pos + 8);
            size_t q2 = block.find("\"", q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos) {
                info.dtype = block.substr(q1 + 1, q2 - q1 - 1);
            }
        }
        
        // Extract shape
        size_t sh_pos = block.find("\"shape\":");
        if (sh_pos != std::string::npos) {
            size_t b1 = block.find("[", sh_pos);
            size_t b2 = block.find("]", b1);
            if (b1 != std::string::npos && b2 != std::string::npos) {
                std::string sh_str = block.substr(b1 + 1, b2 - b1 - 1);
                std::stringstream ss(sh_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) {
                        try {
                            info.shape.push_back(std::stoll(item));
                        } catch (...) {}
                    }
                }
            }
        }
        
        // Extract data_offsets
        size_t do_pos = block.find("\"data_offsets\":");
        if (do_pos != std::string::npos) {
            size_t b1 = block.find("[", do_pos);
            size_t b2 = block.find("]", b1);
            if (b1 != std::string::npos && b2 != std::string::npos) {
                std::string do_str = block.substr(b1 + 1, b2 - b1 - 1);
                std::stringstream ss(do_str);
                std::string s0, s1;
                std::getline(ss, s0, ',');
                std::getline(ss, s1, ',');
                try {
                    info.data_offsets[0] = std::stoull(s0);
                    info.data_offsets[1] = std::stoull(s1);
                } catch (...) {}
            }
        }
        
        // Direct memory pointer
        const char* d_start = mapped_files[file_idx].data_start;
        info.ptr_bf16 = (const uint16_t*)(d_start + info.data_offsets[0]);
        
        tensors[tensor_name] = info;
    }
    
    return true;
}

bool SafetensorsLoader::load_directory(const std::string& dir_path) {
    std::vector<std::string> safetensor_files;
    
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        std::cerr << "[SafetensorsLoader] Error: Cannot open directory " << dir_path << std::endl;
        return false;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fname = entry->d_name;
        if (fname.size() >= 12 && fname.substr(fname.size() - 12) == ".safetensors") {
            safetensor_files.push_back(dir_path + "/" + fname);
        }
    }
    closedir(dir);
    
    std::sort(safetensor_files.begin(), safetensor_files.end());
    
    if (safetensor_files.empty()) {
        std::cerr << "[SafetensorsLoader] Error: No .safetensors files found in " << dir_path << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < safetensor_files.size(); ++i) {
        const auto& fpath = safetensor_files[i];
        int fd = open(fpath.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[SafetensorsLoader] Error: Failed to open " << fpath << std::endl;
            return false;
        }
        
        struct stat st;
        fstat(fd, &st);
        size_t fsize = st.st_size;
        
        void* addr = mmap(NULL, fsize, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, 0);
        if (addr == MAP_FAILED) {
            std::cerr << "[SafetensorsLoader] Error: mmap failed for " << fpath << std::endl;
            close(fd);
            return false;
        }
        
        uint64_t header_len = *(const uint64_t*)addr;
        const char* header_json = (const char*)addr + 8;
        const char* data_start = (const char*)addr + 8 + header_len;
        
        MappedFile mf;
        mf.path = fpath;
        mf.fd = fd;
        mf.addr = addr;
        mf.size = fsize;
        mf.header_len = header_len;
        mf.data_start = data_start;
        
        mapped_files.push_back(mf);
        total_mapped_bytes += fsize;
        
        parse_json_header(static_cast<int>(i), header_json, header_len);
    }
    
    std::cout << "[SafetensorsLoader] Successfully zero-copy mapped " << mapped_files.size() 
              << " shards (" << (total_mapped_bytes / (1024.0*1024.0*1024.0)) << " GB), indexed " 
              << tensors.size() << " tensors." << std::endl;
    return true;
}

const TensorInfo* SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        return &it->second;
    }
    return nullptr;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}

void SafetensorsLoader::prefetch_tensor(const std::string& name) {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        const auto& info = it->second;
        size_t len = info.data_offsets[1] - info.data_offsets[0];
        void* ptr = (void*)info.ptr_bf16;
        madvise(ptr, len, MADV_WILLNEED);
    }
}

void SafetensorsLoader::evict_tensor(const std::string& name) {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        const auto& info = it->second;
        size_t len = info.data_offsets[1] - info.data_offsets[0];
        void* ptr = (void*)info.ptr_bf16;
        madvise(ptr, len, MADV_DONTNEED);
    }
}

} // namespace gemma4
