// kernel-only file: no PyTorch headers here so nvcc doesn't parse them
#include <cuda_runtime.h>

__global__ void mul2_kernel(float* data, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) data[idx] = data[idx] * 2.0f;
}

void attention_mul2_cuda(float* data, size_t numel) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    mul2_kernel<<<blocks, threads>>>(data, numel);
}

