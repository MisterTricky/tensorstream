Excellent question. This moves us from the "what" to the "how" at a professional software design level. Designing `TensorStream` as a separate, installable library (`pip install tensorstream`) is the correct approach for reusability, maintainability, and community contribution.

Here is a detailed architectural design and implementation plan for the `TensorStream` library.

**Timestamp:** Friday, June 13, 2025, 5:56 PM SAST
**Location:** Pretoria, Gauteng, South Africa

---

### **Project Design: The `tensorstream` Library**

**Core Philosophy:** To create a standalone, high-performance Python library for transparently offloading and streaming PyTorch model layers from disk, with a plug-and-play design that requires minimal changes to a user's existing model code.

---

#### **1. High-Level Architecture**

The library will be designed with a clean separation of concerns, consisting of three main layers:

1.  **Public API Layer (The "Plugin Interface"):** This is what the end-user interacts with. It will be a simple, high-level API designed for ease of use.
2.  **Orchestration Layer:** The internal "brain" that manages the state, scheduling, pre-fetching, and I/O pipeline.
3.  **I/O Backend Layer:** The low-level "muscle" that performs the actual data transfers. It will be designed to be modular, with a primary GPUDirect backend and a fallback `mmap` backend.

**Visualized Architecture:**

```
+-------------------------------------------------+
|               User's PyTorch Model              |
| (e.g., Hugging Face MistralForCausalLM)         |
+-------------------------------------------------+
                        |
                        | Uses (via Hooks/Wrapping)
                        v
+-------------------------------------------------+
|        TensorStream Library (Public API)        |
|  `tensorstream.offload(model, config)`          |
+-------------------------------------------------+
                        |
                        | Instantiates & Manages
                        v
+-------------------------------------------------+
|             Orchestration Engine                |
| - Layer State Tracking                          |
| - Pre-fetching & Caching Logic                  |
| - I/O Request Queue                             |
+-------------------------------------------------+
                        |
                        | Selects & Dispatches to
                        v
+-----------------------+-------------------------+
| I/O Backend (Primary) | I/O Backend (Fallback)  |
| - GPUDirect `cuFile`  | - POSIX `mmap`          |
| - (Requires C++/CUDA) | - (Pure Python/CTypes)  |
+-----------------------+-------------------------+
```

---

#### **2. The Public API: `tensorstream.offload()`**

The main design principle is **non-invasiveness**. A user should not have to rewrite their model's `forward` method. We will achieve this using **PyTorch Hooks**.

**Usage Example:**

```python
import torch
import tensorstream
from transformers import AutoModelForCausalLM

# 1. Load the model as usual (it's on the CPU)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# 2. Define TensorStream configuration
config = tensorstream.Config(
    storage_path="/path/to/shards/",
    vram_budget_gb=5,
    backend='auto' # 'auto', 'gpudirect', or 'mmap'
)

# 3. Apply TensorStream - The "Magic" happens here
# This function returns a modified model object
offloaded_model = tensorstream.offload(model, config)

# 4. Use the model as normal
offloaded_model.to('cuda:0')
output = offloaded_model.generate(...)
```

**How `tensorstream.offload()` Works Internally (The Plugin Mechanism):**

The `offload()` function is the heart of the plugin design. It will:
1.  **Instantiate the Orchestrator:** Create an instance of the `OrchestrationEngine` with the given config.
2.  **Shard the Model:** Iterate through the `model.layers` and save each one to disk in the `.ts` format using the specified `storage_path`.
3.  **Replace Layers with Proxies:** Iterate through `model.layers` again. For each layer (e.g., `model.layers[i]`), it will replace the `nn.Module` object with a lightweight **`TensorStreamProxyLayer`** object.
4.  **Register Forward Hooks:** This is the key to non-invasive operation. The `TensorStreamProxyLayer` will have a `register_forward_pre_hook`. This hook is a function that PyTorch automatically calls *right before* the layer's `forward` method is executed.
    * The hook function will call the `OrchestrationEngine` and say, "I am layer `i`, I am about to be executed. Please ensure my weights are in VRAM."
    * The Orchestrator will then block until the I/O backend has loaded the weights for layer `i`. Once loaded, the real `forward` pass can proceed.

This hook-based design means the user's `generate()` call works without modification, but behind the scenes, `TensorStream` is intercepting the execution flow to stream data just-in-time.

---

#### **3. Detailed Implementation Plan**

This plan outlines the creation of the installable `tensorstream` library.

##### **Phase 1: Project Scaffolding & Core Utilities (2 Weeks)**

* **Task 1.1: Library Structure**
    * **Action:** Set up a standard Python library project structure using `pyproject.toml` and `setup.py`.
    * **Key Files:** `pyproject.toml`, `setup.py`, `tensorstream/__init__.py`, `tensorstream/api.py`, `tensorstream/orchestrator.py`, `tensorstream/backends/`.

* **Task 1.2: Serializer & File Format**
    * **Action:** Implement the `save_to_ts(tensor, path)` and `read_ts_header(path)` functions. This is a pure Python task.
    * **Deliverable:** A utility module `tensorstream/io.py` that can reliably read and write the `.ts` file format.

* **Task 1.3: The `mmap` Fallback Backend**
    * **Action:** Create `tensorstream/backends/mmap_backend.py`. This class will have `load(path)` and `unload(handle)` methods that use Python's built-in `mmap` and `ctypes` to interact with CUDA's memory APIs.
    * **Deliverable:** A fully functional, albeit slower, I/O backend.

##### **Phase 2: The GPUDirect Backend (4 Weeks)**

* **Task 2.1: C++/CUDA Extension for `cuFile`**
    * **Action:** This is a low-level task. Create a C++ source file that includes the `cuFile.h` header. Write wrapper functions for the core GPUDirect calls (e.g., `cufile_read_wrapper`).
    * **Key Files:** `tensorstream/csrc/gds.cpp`.

* **Task 2.2: Python Bindings**
    * **Action:** Use **Pybind11** to expose the C++ wrapper functions to Python. This creates a bridge between the low-level code and the Python orchestrator.
    * **Deliverable:** A compiled `.so` file that can be imported in Python.

* **Task 2.3: Packaging the Extension**
    * **Action:** Modify `setup.py` to use `CUDAExtension` from `torch.utils.cpp_extension` or a similar tool (`setuptools-cuda`). This will tell `pip` how to compile the C++/CUDA code during the library's installation process.
    * **Deliverable:** The ability to successfully run `pip install .` and have the GPUDirect extension compile and be ready for use.

##### **Phase 3: The API & Orchestration Layer (3 Weeks)**

* **Task 3.1: The Orchestration Engine**
    * **Action:** Implement the `OrchestrationEngine` class. It will manage the layer state ("on disk", "in RAM", "in VRAM"), the request queue, and the background I/O thread pool. At initialization, it will try to import the GPUDirect backend and fall back to the `mmap` backend if it fails.
    * **Deliverable:** The core logic engine of the library.

* **Task 3.2: Proxy Layers and Hooks**
    * **Action:** Implement the `TensorStreamProxyLayer` class. Its main purpose is to hold metadata about the real layer (e.g., its path on disk) and to host the `register_forward_pre_hook`.

* **Task 3.3: The Public API**
    * **Action:** Implement the main `tensorstream.offload(model, config)` function in `tensorstream/api.py`. This function will perform the sharding and hook registration loop.
    * **Deliverable:** The final, user-facing entry point of the library.

##### **Phase 4: Testing, Documentation, and Packaging (2 Weeks)**

* **Task 4.1: Rigorous Testing**
    * **Action:** Test the full pipeline with a real model like Mistral-7B. Verify that it works correctly with both the `mmap` and `GPUDirect` backends. Profile the performance difference.

* **Task 4.2: Documentation**
    * **Action:** Write comprehensive documentation, including installation instructions (with GPUDirect prerequisites), API reference, and usage examples.

* **Task 4.3: Packaging for PyPI**
    * **Action:** Create the necessary package files (`sdist`, `wheel`) and upload to the Python Package Index (PyPI) for easy installation via `pip`.
    * **Deliverable:** The publicly available `tensorstream` library.