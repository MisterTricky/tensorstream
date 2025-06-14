/*
 * GPUDirect Storage wrapper header
 */

#pragma once

#ifdef WITH_GDS
#include <cufile.h>
#endif

#include <string>
#include <vector>

namespace tensorstream {

struct FileHandle {
#ifdef WITH_GDS
    CUfileHandle_t cufile_handle;
    int fd;
#else
    void* dummy;
    int fd;
#endif
};

class GDSWrapper {
public:
    GDSWrapper();
    ~GDSWrapper();
    
    bool initialize();
    void cleanup();
    
    FileHandle open_file(const std::string& path);
    void close_file(const FileHandle& handle);
    
    size_t read_direct(const FileHandle& handle, void* gpu_ptr, 
                      size_t size, size_t offset);
    
    void register_buffer(void* gpu_ptr, size_t size);
    void unregister_buffer(void* gpu_ptr);
    
private:
    bool initialized_;
#ifdef WITH_GDS
    std::vector<CUfileHandle_t> file_handles_;
#endif
};

}  // namespace tensorstream
