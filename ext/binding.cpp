#include <torch/extension.h>

// Forward declarations of the CUDA wrapper functions implemented in the source files
at::Tensor attention_mul2(at::Tensor input);
at::Tensor patch_embed(at::Tensor img, at::Tensor weights);

PYBIND11_MODULE(vit_cuda, m) {
    m.doc() = "vit_cuda extension (CUDA backend placeholder)";
    m.def("attention_mul2", &attention_mul2, "Multiply tensor by 2 on GPU");
    m.def("patch_embed", &patch_embed, "Patch embedding (CUDA)");
}
