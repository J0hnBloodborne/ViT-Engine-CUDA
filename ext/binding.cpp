#include <torch/extension.h>

// Forward declarations
at::Tensor flash_attn_2(at::Tensor Q, at::Tensor K, at::Tensor V, float scale);
at::Tensor patch_embed(at::Tensor img, at::Tensor weights);
at::Tensor pos_encoding(at::Tensor patches, at::Tensor cls_token, at::Tensor pos_embeddings);

PYBIND11_MODULE(vit_cuda, m) {
    m.doc() = "vit_cuda extension (Flash Attention 2 implemented)";
    m.def("flash_attn_2", &flash_attn_2, "Flash Attention 2 forward pass");
    m.def("patch_embed", &patch_embed, "Patch embedding (CUDA)");
    m.def("pos_encoding", &pos_encoding, "Add positional encoding (CUDA)");
}
