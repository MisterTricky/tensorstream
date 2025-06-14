/*
 * GPUDirect Storage core implementation for TensorStream
 * 
 * Provides direct storage-to-GPU transfers using NVIDIA's GPUDirect Storage.
 */

#ifdef WITH_GDS

#include <torch/extension.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>

// File handle tracking
static std::unordered_map<std::string, CUfileHandle_t> g_file_handles;
static std::unordered_map<void*, CUdeviceptr> g_memory_registrations;
static std::mutex g_gds_mutex;
static bool g_gds_initialized = false;

// Initialize GPUDirect Storage
bool initialize_gds() {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    if (g_gds_initialized) {
        return true;
    }
    
    try {
        // Initialize cuFile
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            return false;
        }
        
        g_gds_initialized = true;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// Cleanup GPUDirect Storage
void cleanup_gds() {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    if (!g_gds_initialized) {
        return;
    }
    
    // Close all file handles
    for (auto& pair : g_file_handles) {
        cuFileHandleDeregister(pair.second);
    }
    g_file_handles.clear();
    
    // Unregister all memory
    for (auto& pair : g_memory_registrations) {
        cuMemFree(pair.second);
    }
    g_memory_registrations.clear();
    
    // Close cuFile driver
    cuFileDriverClose();
    g_gds_initialized = false;
}

// Open file for GDS operations
int64_t open_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    if (!g_gds_initialized) {
        throw std::runtime_error("GDS not initialized");
    }
    
    // Check if already open
    auto it = g_file_handles.find(file_path);
    if (it != g_file_handles.end()) {
        return reinterpret_cast<int64_t>(it->second.fh);
    }
    
    // Open file
    CUfileDescr_t cf_descr;
    cf_descr.handle.fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (cf_descr.handle.fd < 0) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    // Register with cuFile
    CUfileHandle_t cf_handle;
    CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(cf_descr.handle.fd);
        throw std::runtime_error("Failed to register file with cuFile");
    }
    
    g_file_handles[file_path] = cf_handle;
    return reinterpret_cast<int64_t>(cf_handle.fh);
}

// Close file
void close_file(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    // Find and remove file handle
    for (auto it = g_file_handles.begin(); it != g_file_handles.end(); ++it) {
        if (reinterpret_cast<int64_t>(it->second.fh) == handle) {
            cuFileHandleDeregister(it->second);
            g_file_handles.erase(it);
            break;
        }
    }
}

// Register GPU memory for GDS
int64_t register_memory(int64_t gpu_ptr, int64_t size) {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    void* ptr = reinterpret_cast<void*>(gpu_ptr);
    
    // Check if already registered
    auto it = g_memory_registrations.find(ptr);
    if (it != g_memory_registrations.end()) {
        return reinterpret_cast<int64_t>(it->second);
    }
    
    // Register memory with cuFile
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr);
    CUfileError_t status = cuFileBufRegister(dev_ptr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("Failed to register memory with cuFile");
    }
    
    g_memory_registrations[ptr] = dev_ptr;
    return reinterpret_cast<int64_t>(dev_ptr);
}

// Unregister GPU memory
void unregister_memory(int64_t registration) {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(registration);
    
    // Find and remove registration
    for (auto it = g_memory_registrations.begin(); it != g_memory_registrations.end(); ++it) {
        if (it->second == dev_ptr) {
            cuFileBufDeregister(dev_ptr);
            g_memory_registrations.erase(it);
            break;
        }
    }
}

// Read data directly from storage to GPU
int64_t read_direct(int64_t file_handle, int64_t file_offset, 
                   int64_t gpu_ptr, int64_t size) {
    // Convert handles back to cuFile types
    CUfileHandle_t cf_handle;
    cf_handle.fh = reinterpret_cast<void*>(file_handle);
    
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(gpu_ptr);
    
    // Perform direct read
    ssize_t bytes_read = cuFileRead(cf_handle, reinterpret_cast<void*>(dev_ptr), 
                                   size, file_offset, 0);
    
    if (bytes_read < 0) {
        throw std::runtime_error("cuFileRead failed");
    }
    
    return static_cast<int64_t>(bytes_read);
}

// Get GDS statistics
std::unordered_map<std::string, int64_t> get_stats() {
    std::lock_guard<std::mutex> lock(g_gds_mutex);
    
    std::unordered_map<std::string, int64_t> stats;
    stats["open_files"] = static_cast<int64_t>(g_file_handles.size());
    stats["registered_buffers"] = static_cast<int64_t>(g_memory_registrations.size());
    stats["initialized"] = g_gds_initialized ? 1 : 0;
    
    return stats;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TensorStream GPUDirect Storage Extension";
    
    m.def("initialize_gds", &initialize_gds, "Initialize GPUDirect Storage");
    m.def("cleanup_gds", &cleanup_gds, "Cleanup GPUDirect Storage");
    m.def("open_file", &open_file, "Open file for GDS operations");
    m.def("close_file", &close_file, "Close GDS file");
    m.def("register_memory", &register_memory, "Register GPU memory for GDS");
    m.def("unregister_memory", &unregister_memory, "Unregister GPU memory");
    m.def("read_direct", &read_direct, "Read directly from storage to GPU");
    m.def("get_stats", &get_stats, "Get GDS statistics");
}

#else  // !WITH_GDS

// Stub implementation when GDS is not available
#include <torch/extension.h>
#include <stdexcept>

bool initialize_gds() {
    throw std::runtime_error("GPUDirect Storage not available - compiled without GDS support");
}

void cleanup_gds() {}
int64_t open_file(const std::string& file_path) {
    throw std::runtime_error("GPUDirect Storage not available");
}
void close_file(int64_t handle) {}
int64_t register_memory(int64_t gpu_ptr, int64_t size) {
    throw std::runtime_error("GPUDirect Storage not available");
}
void unregister_memory(int64_t registration) {}
int64_t read_direct(int64_t file_handle, int64_t file_offset, int64_t gpu_ptr, int64_t size) {
    throw std::runtime_error("GPUDirect Storage not available");
}
std::unordered_map<std::string, int64_t> get_stats() {
    return {{"available", 0}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TensorStream GPUDirect Storage Extension (Stub)";
    
    m.def("initialize_gds", &initialize_gds, "Initialize GPUDirect Storage (stub)");
    m.def("cleanup_gds", &cleanup_gds, "Cleanup GPUDirect Storage (stub)");
    m.def("open_file", &open_file, "Open file for GDS operations (stub)");
    m.def("close_file", &close_file, "Close GDS file (stub)");
    m.def("register_memory", &register_memory, "Register GPU memory for GDS (stub)");
    m.def("unregister_memory", &unregister_memory, "Unregister GPU memory (stub)");
    m.def("read_direct", &read_direct, "Read directly from storage to GPU (stub)");
    m.def("get_stats", &get_stats, "Get GDS statistics (stub)");
}

#endif  // WITH_GDS
