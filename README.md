# Hopper-Ultra-GEMM 🚀

**Hopper-Ultra-GEMM** is an extreme-performance matrix multiplication (GEMM) engine specifically handcrafted for the **NVIDIA Hopper (sm_90a)** architecture. This project pushes the boundaries of GPU computing by integrating next-generation hardware features to achieve maximum throughput for AI workloads.

## 🛠 Architectural Innovations

This implementation moves beyond traditional CUDA kernels by leveraging the unique hardware blocks of the H100:

### 1. Thread Block Clusters & DSMEM
Utilizes the new **Thread Block Cluster** programming model to enable **Distributed Shared Memory (DSMEM)**. 
- Blocks are organized into a 2x2x2 cluster, allowing threads to directly read/write to the Shared Memory of peer blocks within the cluster.
- Reduces global memory pressure and enables ultra-low latency data exchange.

### 2. Hardware Tensor Memory Accelerator (TMA)
Bypasses traditional software-based data movement. 
- Uses asynchronous **TMA descriptors** to fetch 3D tensor tiles directly into Shared Memory.
- Offloads address calculation and boundary checking to dedicated hardware logic, freeing up Warp execution resources.

### 3. Advanced Warp Specialization
Implements a sophisticated **Producer-Consumer** model by partitioning warps into specialized roles:
- **TMA_CONTROLLER:** Manages asynchronous hardware descriptors and triggers data movement.
- **MATH_COMPUTE:** Dedicated exclusively to driving the **Tensor Cores** for `mma.sync` operations.
- **STORE_EPILOGUE:** Handles real-time FP8 quantization, scaling, and coalesced global memory write-back.

### 4. Multi-Stage Software Pipelining
Features a 3-stage asynchronous pipeline (Load -> Compute -> Store) synchronized via hardware **mbarrier** objects. This architecture completely hides memory latency behind the massive compute throughput of the Tensor Cores.

## 📊 Technical Specifications

- **Precision:** FP8 (`e4m3`) accumulation with FP32 feedback.
- **Architecture:** Optimized for NVIDIA sm_90a (Hopper).
- **Instruction Set:** Leveraging PTX `cp.async.bulk` and `mma.sync.aligned.m16n8k16.f32.e4m3.e4m3.f32`.
- **Memory Management:** 128-byte aligned memory arenas for zero-copy TMA compatibility.

## 🏗 Build and Requirements

### Prerequisites
- **Hardware:** NVIDIA H100 GPU or newer.
- **Toolkit:** CUDA Toolkit 12.0 or higher.
- **Compiler:** `nvcc`

### Compilation
To compile the kernel for the Hopper architecture, use the following command:
```bash
nvcc -arch=sm_90a -lcuda main.cu -o hopper_gemm
