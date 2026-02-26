# CS5463-GPU-GEMM-Optimization
## Term Project: GPU GEMM Optimization

### Student: Thanush Koshekay

## Problem Statement
CUDA-based optimization and performance analysis of GPU matrix multiplication (GEMM) for modern ML workloads.

Proposed Topic: Matrix Multiplication Optimization of High-Performance GPU to serve Modern ML Workloads

One of the most fundamental computations in high-performance computing systems, as well as in present-day machine learning, is called matrix multiplication (GEMM). A wide range of workloads of deep learning, such as fully connected layers, convolutional layers (with reduction), and attention mechanisms, are dependent on dense matrix multiplications extensively. Due to this reason, the optimization of GEMM in contemporary GPU hardware continues to be an important performance issue. This project is aimed at implementing and testing various strategies of matrix multiplication based on CUDA and searching for ways of how optimization can be made architectural aware and enhance the performance accordingly.

This project will use three competitive approaches of computing by GMUs. C=AxB: (1) a naive implementation using global memory accesses and each thread computing one output element is a benchmark implementation of a baseline implementation, (2) an implementation using shared memory accesses each thread computing a single output element is an implementation designed to mitigate global memory accesses and enhance data reuse, and (3) an implementation applying more advanced methods like register blocking and loop unrolling to further multiply arithmetic intensity and reduce the amount of synchronization in the code. They will be contrasted with a CPU reference as well as the cuBLAS library of NVIDIA to evaluate the performance efficiency alongside optimized routines in the industry.

A GPU cluster will be analyzed in terms of performance through CUDA event-based timing in different sizes of matrices (e.g., 256x256 until 4096x4096). Measures will consist of the execution time (ms), throughput (GFLOP/s), as well as the speedup against the middle- ground implementations. Validity will be validated by means of comparing numbers with a credible reference implementation. The final deliverable will be the performance plots and memory access behavior, synchronization behavior analysis, and trade-off between the complexity of implementation and rate of optimization. The project is expected to provide individual insight into the practical understanding of the memory hierarchy of the GPUs, parallel workload mapping, as well as the strategies of optimization of the accelerator-based computing.
<img width="468" height="607" alt="image" src="https://github.com/user-attachments/assets/b038bf45-2177-4d62-9ed8-7d6777ba795a" />
