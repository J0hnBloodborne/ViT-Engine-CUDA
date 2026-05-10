#include <torch/extension.h>

void flash_attn_2_forward_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, float scale
);

at::Tensor flash_attn_2(at::Tensor Q, at::Tensor K, at::Tensor V, float scale) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    // Q, K, V are expected to be [B, N, 768]
    TORCH_CHECK(Q.dim() == 3, "Expected 3D tensor [B, N, 768]");
    
    int B = Q.size(0);
    int N = Q.size(1);
    int E = Q.size(2);
    
    TORCH_CHECK(E == 768, "E dimension must be 768");

    auto O = at::zeros_like(Q);

    flash_attn_2_forward_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, N, scale
    );

    return O;
}
