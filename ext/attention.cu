// testing a simple CUDA extension in PyTorch
#include <torch/extension.h>

__global__ void mul2_kernel(float* data, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) data[idx] = data[idx] * 2.0f;
}

// C++ wrapper called from binding.cpp / pybind11
at::Tensor attention_mul2(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    at::Tensor out = input.contiguous();
    size_t numel = out.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mul2_kernel<<<blocks, threads>>>(out.data_ptr<float>(), numel);
    return out;
}

