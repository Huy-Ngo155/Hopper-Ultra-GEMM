# Hopper-Ultra-GEMM 🚀

**Hopper-Ultra-GEMM** is a high-performance matrix multiplication engine specifically designed to leverage the cutting-edge features of the **NVIDIA Hopper (H100)** architecture. This project represents the pinnacle of CUDA optimization, moving beyond traditional kernels to use hardware-accelerated memory management and inter-block communication.

## 🛠 Advanced Features

* **Thread Block Clusters:** Utilizes `cluster.sync` and `distributed shared memory` to allow blocks within a cluster to share data directly, bypassing global memory.
* **TMA (Tensor Memory Accelerator):** Implements asynchronous bulk data movement using hardware TMA descriptors, reducing CPU overhead and improving memory bandwidth utilization.
* **Warp Specialization:** Partitioned warps into specialized roles: **TMA_CONTROLLER**, **MATH_COMPUTE**, and **STORE_EPILOGUE** to maximize instruction throughput.
* **FP8 Precision & MMA:** Full support for `e4m3` FP8 formats using `mma.sync.aligned.m16n8k16` instructions for massive throughput in AI workloads.
* **MBarrier Synchronization:** Low-level synchronization using hardware `mbarrier` objects for efficient producer-consumer pipelining.

## 🏗 Hardware Requirements
- **Architecture:** NVIDIA Hopper (sm_90a) or newer.
- **Compiler:** NVCC 12.0+ with CUDA Toolkit.
- **Features:** Requires hardware support for TMA and Distributed Shared Memory.

## 🚀 Technical Highlights
- **DSMEM Data Exchange:** Direct remote shared memory access across blocks within a 2x2x2 cluster.
- **Software Pipelining:** 3-stage asynchronous pipeline (Load -> Compute -> Store) to hide memory latency.
- **Manual Quantization:** Built-in FP8 quantization and scaling logic in the epilogue phase.
