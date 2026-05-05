#include <torch/extension.h>

// Forward declaration of the CUDA kernel wrapper implemented in attention.cu
at::Tensor attention_mul2(at::Tensor input);

PYBIND11_MODULE(vit_cuda, m) {
    m.doc() = "vit_cuda extension (CUDA backend placeholder)";
    m.def("attention_mul2", &attention_mul2, "Multiply tensor by 2 on GPU");
}
