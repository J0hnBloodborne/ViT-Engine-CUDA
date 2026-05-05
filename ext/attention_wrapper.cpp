#include <torch/extension.h>

void attention_mul2_cuda(float* data, size_t numel);

at::Tensor attention_mul2(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    at::Tensor out = input.contiguous();
    size_t numel = out.numel();
    attention_mul2_cuda(out.data_ptr<float>(), numel);
    return out;
}
