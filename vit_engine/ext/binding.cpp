#include <torch/extension.h>

PYBIND11_MODULE(vit_cuda, m) {
    m.doc() = "vit_cuda extension (CUDA backend placeholder)";
}
