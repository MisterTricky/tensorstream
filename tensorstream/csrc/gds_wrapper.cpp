/*
 * GPUDirect Storage wrapper functions
 * 
 * Provides clean C++ wrappers around cuFile operations.
 */

#ifdef WITH_GDS

#include "gds_wrapper.h"
#include <cufile.h>
#include <stdexcept>
#include <cstring>

namespace tensorstream {

GDSWrapper::GDSWrapper() : initialized_(false) {}

GDSWrapper::~GDSWrapper() {
    if (initialized_) {
        cleanup();
    }
}

bool GDSWrapper::initialize() {
    if (initialized_) {
        return true;
    }
    
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

void GDSWrapper::cleanup() {
    if (!initialized_) {
        return;
    }
    
    // Close all file handles
    for (auto& handle : file_handles_) {
        cuFileHandleDeregister(handle);
    }
    file_handles_.clear();
    
    cuFileDriverClose();
    initialized_ = false;
}

FileHandle GDSWrapper::open_file(const std::string& path) {
    if (!initialized_) {
        throw std::runtime_error("GDS not initialized");
    }
    
    CUfileDescr_t cf_descr;
    std::memset(&cf_descr, 0, sizeof(cf_descr));
    
    cf_descr.handle.fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (cf_descr.handle.fd < 0) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileHandle_t cf_handle;
    CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(cf_descr.handle.fd);
        throw std::runtime_error("Failed to register file with cuFile");
    }
    
    file_handles_.push_back(cf_handle);
    return FileHandle{cf_handle, cf_descr.handle.fd};
}

void GDSWrapper::close_file(const FileHandle& handle) {
    auto it = std::find(file_handles_.begin(), file_handles_.end(), handle.cufile_handle);
    if (it != file_handles_.end()) {
        cuFileHandleDeregister(*it);
        close(handle.fd);
        file_handles_.erase(it);
    }
}

size_t GDSWrapper::read_direct(const FileHandle& handle, void* gpu_ptr, 
                              size_t size, size_t offset) {
    ssize_t bytes_read = cuFileRead(handle.cufile_handle, gpu_ptr, size, offset, 0);
    if (bytes_read < 0) {
        throw std::runtime_error("cuFileRead failed");
    }
    return static_cast<size_t>(bytes_read);
}

void GDSWrapper::register_buffer(void* gpu_ptr, size_t size) {
    CUfileError_t status = cuFileBufRegister(reinterpret_cast<CUdeviceptr>(gpu_ptr), size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("Failed to register buffer with cuFile");
    }
}

void GDSWrapper::unregister_buffer(void* gpu_ptr) {
    cuFileBufDeregister(reinterpret_cast<CUdeviceptr>(gpu_ptr));
}

}  // namespace tensorstream

#else  // !WITH_GDS

#include "gds_wrapper.h"
#include <stdexcept>

namespace tensorstream {

GDSWrapper::GDSWrapper() : initialized_(false) {}
GDSWrapper::~GDSWrapper() {}

bool GDSWrapper::initialize() {
    return false;  // GDS not available
}

void GDSWrapper::cleanup() {}

FileHandle GDSWrapper::open_file(const std::string& path) {
    throw std::runtime_error("GPUDirect Storage not available");
}

void GDSWrapper::close_file(const FileHandle& handle) {}

size_t GDSWrapper::read_direct(const FileHandle& handle, void* gpu_ptr, 
                              size_t size, size_t offset) {
    throw std::runtime_error("GPUDirect Storage not available");
}

void GDSWrapper::register_buffer(void* gpu_ptr, size_t size) {
    throw std::runtime_error("GPUDirect Storage not available");
}

void GDSWrapper::unregister_buffer(void* gpu_ptr) {}

}  // namespace tensorstream

#endif  // WITH_GDS
