/*
 * Memory Manager header for TensorStream
 * 
 * Provides advanced memory management for CUDA operations.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <set>
#include <mutex>
#include <cstddef>

namespace tensorstream {

struct MemoryBlock {
    size_t size;
    bool in_use;
    uint64_t request_id;
};

struct MemoryStats {
    size_t pool_size;
    size_t allocated_bytes;
    size_t free_bytes;
    size_t num_allocations;
    size_t num_free_blocks;
};

class MemoryManager {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 256;
    static constexpr size_t MIN_BLOCK_SIZE = 1024;
    
    explicit MemoryManager(size_t pool_size = 1024 * 1024 * 1024);  // 1GB default
    ~MemoryManager();
    
    // Non-copyable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    bool initialize();
    void cleanup();
    
    void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT);
    void deallocate(void* ptr);
    
    MemoryStats get_stats() const;
    
private:
    mutable std::mutex mutex_;
    bool initialized_ = false;
    
    size_t pool_size_;
    size_t allocated_bytes_;
    uint64_t next_request_id_ = 1;
    
    // Memory tracking
    std::unordered_map<void*, MemoryBlock> memory_blocks_;
    std::set<std::pair<size_t, void*>> free_blocks_;  // size -> ptr mapping
    
    void* allocate_new_block(size_t size);
    size_t align_size(size_t size, size_t alignment) const;
    void coalesce_blocks(void* ptr);
};

}  // namespace tensorstream
