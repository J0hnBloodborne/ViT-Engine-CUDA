#include <torch/extension.h>
#include <vector>

void mlp_forward_cuda(
    const float* X, const float* W1, const float* B1, 
    const float* W2, const float* B2, 
    float* H, float* O, 
    int B, int N, int E, int E_expand
);

std::vector<at::Tensor> mlp_forward(
    at::Tensor X, 
    at::Tensor W1, at::Tensor B1, 
    at::Tensor W2, at::Tensor B2
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W1.is_cuda(), "W1 must be a CUDA tensor");
    TORCH_CHECK(B1.is_cuda(), "B1 must be a CUDA tensor");
    TORCH_CHECK(W2.is_cuda(), "W2 must be a CUDA tensor");
    TORCH_CHECK(B2.is_cuda(), "B2 must be a CUDA tensor");

    TORCH_CHECK(X.dim() == 3, "X must be 3D");

    int B = X.size(0);
    int N = X.size(1);
    int E = X.size(2);
    int E_expand = W1.size(0);

    TORCH_CHECK(W1.size(1) == E, "W1 shape mismatch");
    TORCH_CHECK(W2.size(0) == E, "W2 shape mismatch");
    TORCH_CHECK(W2.size(1) == E_expand, "W2 shape mismatch");

    auto H = at::empty({B, N, E_expand}, X.options());
    auto O = at::empty({B, N, E}, X.options());

    mlp_forward_cuda(
        X.data_ptr<float>(),
        W1.data_ptr<float>(), B1.data_ptr<float>(),
        W2.data_ptr<float>(), B2.data_ptr<float>(),
        H.data_ptr<float>(), O.data_ptr<float>(),
        B, N, E, E_expand
    );

    return {O, H};
}
