#include <torch/extension.h>

void flash_attn_2_forward_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int D, float scale
);

at::Tensor flash_attn_2(at::Tensor Q, at::Tensor K, at::Tensor V, float scale) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    auto O = at::zeros_like(Q);

    flash_attn_2_forward_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D, scale
    );

    return O;
}
