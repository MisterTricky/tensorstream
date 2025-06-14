#!/usr/bin/env python3
"""
Setup script for TensorStream library.
Handles both pure Python components and optional CUDA/GPUDirect extensions.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup


def find_cuda_home() -> Optional[str]:
    """Find CUDA installation directory."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return cuda_home
    
    # Try common CUDA installation paths
    for path in ["/usr/local/cuda", "/opt/cuda", "/usr/cuda"]:
        if os.path.isdir(path):
            return path
    
    # Try to find via nvcc
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], text=True).strip()
        return str(Path(nvcc_path).parent.parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return None


def check_gds_support() -> bool:
    """Check if GPUDirect Storage (GDS) is available."""
    cuda_home = find_cuda_home()
    if not cuda_home:
        return False
    
    # Check for cuFile header
    cufile_header = Path(cuda_home) / "include" / "cufile.h"
    if not cufile_header.exists():
        print("Warning: cufile.h not found. GPUDirect Storage backend will be disabled.")
        return False
    
    # Check for cuFile library
    lib_paths = [
        Path(cuda_home) / "lib64" / "libcufile.so",
        Path(cuda_home) / "lib" / "libcufile.so",
    ]
    
    for lib_path in lib_paths:
        if lib_path.exists():
            return True
    
    print("Warning: libcufile.so not found. GPUDirect Storage backend will be disabled.")
    return False


def get_cuda_extensions() -> List[Extension]:
    """Build CUDA extensions if available."""
    extensions = []
    
    cuda_home = find_cuda_home()
    if not cuda_home:
        print("CUDA not found. Building without GPU acceleration.")
        return extensions
    
    print(f"Found CUDA at: {cuda_home}")
    
    # Common CUDA compilation flags
    cuda_flags = [
        "-std=c++14",
        "-O3",
        "-DWITH_CUDA",
        "-DTORCH_EXTENSION_NAME=tensorstream_cuda",
    ]
    
    include_dirs = [
        f"{cuda_home}/include",
        get_cmake_dir() + "/../include",  # pybind11 headers
    ]
    
    library_dirs = [
        f"{cuda_home}/lib64",
        f"{cuda_home}/lib",
    ]
    
    libraries = ["cuda", "cudart"]
    
    # Core CUDA extension
    cuda_ext = Pybind11Extension(
        "tensorstream._cuda_core",
        sources=[
            "tensorstream/csrc/cuda_core.cpp",
            "tensorstream/csrc/memory_manager.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        cxx_std=14,
        extra_compile_args=cuda_flags,
    )
    extensions.append(cuda_ext)
    
    # GPUDirect Storage extension (optional)
    if check_gds_support():
        print("Building with GPUDirect Storage support.")
        gds_flags = cuda_flags + ["-DWITH_GDS"]
        gds_libraries = libraries + ["cufile"]
        
        gds_ext = Pybind11Extension(
            "tensorstream._gds_core",
            sources=[
                "tensorstream/csrc/gds_core.cpp",
                "tensorstream/csrc/gds_wrapper.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=gds_libraries,
            cxx_std=14,
            extra_compile_args=gds_flags,
        )
        extensions.append(gds_ext)
    else:
        print("Building without GPUDirect Storage support.")
    
    return extensions


class TensorStreamBuildExt(build_ext):
    """Custom build extension that handles CUDA compilation gracefully."""
    
    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as e:
            if "cuda" in ext.name.lower() or "gds" in ext.name.lower():
                print(f"Warning: Failed to build {ext.name}: {e}")
                print("Continuing without GPU acceleration...")
            else:
                raise


if __name__ == "__main__":
    # Get extensions
    ext_modules = get_cuda_extensions()
    
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": TensorStreamBuildExt},
        zip_safe=False,
    )
