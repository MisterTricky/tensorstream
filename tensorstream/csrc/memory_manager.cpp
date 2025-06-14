/*
 * Memory Manager for TensorStream CUDA operations
 * 
 * Provides advanced memory management capabilities including
 * memory pooling, alignment, and efficient allocation strategies.
 */

#include "memory_manager.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace tensorstream {

MemoryManager::MemoryManager(size_t pool_size) 
    : pool_size_(pool_size), allocated_bytes_(0) {
}

MemoryManager::~MemoryManager() {
    cleanup();
}

bool MemoryManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true;
    }
    
    try {
        // Allocate initial memory pool
        void* pool_ptr;
        cudaError_t error = cudaMalloc(&pool_ptr, pool_size_);
        if (error != cudaSuccess) {
            return false;
        }
        
        // Initialize memory blocks
        memory_blocks_[pool_ptr] = {pool_size_, false, 0};
        free_blocks_.insert({pool_size_, pool_ptr});
        
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void MemoryManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Free all allocated blocks
    for (auto& pair : memory_blocks_) {
        cudaFree(pair.first);
    }
    
    memory_blocks_.clear();
    free_blocks_.clear();
    allocated_bytes_ = 0;
    initialized_ = false;
}

void* MemoryManager::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    
    // Align size
    size_t aligned_size = align_size(size, alignment);
    
    // Find suitable free block
    auto it = free_blocks_.lower_bound({aligned_size, nullptr});
    if (it == free_blocks_.end()) {
        // No suitable block found, try to allocate new memory
        return allocate_new_block(aligned_size);
    }
    
    void* ptr = it->second;
    size_t block_size = it->first;
    
    // Remove from free blocks
    free_blocks_.erase(it);
    
    // Update block info
    auto& block_info = memory_blocks_[ptr];
    block_info.in_use = true;
    block_info.request_id = next_request_id_++;
    
    // Split block if necessary
    if (block_size > aligned_size + MIN_BLOCK_SIZE) {
        void* remaining_ptr = static_cast<char*>(ptr) + aligned_size;
        size_t remaining_size = block_size - aligned_size;
        
        memory_blocks_[remaining_ptr] = {remaining_size, false, 0};
        free_blocks_.insert({remaining_size, remaining_ptr});
        
        // Update current block size
        block_info.size = aligned_size;
    }
    
    allocated_bytes_ += aligned_size;
    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = memory_blocks_.find(ptr);
    if (it == memory_blocks_.end() || !it->second.in_use) {
        return;  // Invalid pointer or already freed
    }
    
    auto& block_info = it->second;
    block_info.in_use = false;
    block_info.request_id = 0;
    
    allocated_bytes_ -= block_info.size;
    
    // Add to free blocks
    free_blocks_.insert({block_info.size, ptr});
    
    // Try to coalesce with adjacent blocks
    coalesce_blocks(ptr);
}

MemoryStats MemoryManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MemoryStats stats;
    stats.pool_size = pool_size_;
    stats.allocated_bytes = allocated_bytes_;
    stats.free_bytes = pool_size_ - allocated_bytes_;
    stats.num_allocations = 0;
    stats.num_free_blocks = free_blocks_.size();
    
    for (const auto& pair : memory_blocks_) {
        if (pair.second.in_use) {
            stats.num_allocations++;
        }
    }
    
    return stats;
}

void* MemoryManager::allocate_new_block(size_t size) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + 
                                std::string(cudaGetErrorString(error)));
    }
    
    memory_blocks_[ptr] = {size, true, next_request_id_++};
    allocated_bytes_ += size;
    
    return ptr;
}

size_t MemoryManager::align_size(size_t size, size_t alignment) const {
    return (size + alignment - 1) & ~(alignment - 1);
}

void MemoryManager::coalesce_blocks(void* ptr) {
    // This is a simplified coalescing implementation
    // A full implementation would track adjacent blocks and merge them
    // when they become free to reduce fragmentation
    
    // For now, we just ensure the block is marked as free
    auto it = memory_blocks_.find(ptr);
    if (it != memory_blocks_.end()) {
        it->second.in_use = false;
    }
}

}  // namespace tensorstream
