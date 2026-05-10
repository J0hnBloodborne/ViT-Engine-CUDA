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
    at::Tensor X_c = X.contiguous();
    at::Tensor W1_c = W1.contiguous();
    at::Tensor B1_c = B1.contiguous();
    at::Tensor W2_c = W2.contiguous();
    at::Tensor B2_c = B2.contiguous();

    TORCH_CHECK(X_c.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W1_c.is_cuda(), "W1 must be a CUDA tensor");
    TORCH_CHECK(B1_c.is_cuda(), "B1 must be a CUDA tensor");
    TORCH_CHECK(W2_c.is_cuda(), "W2 must be a CUDA tensor");
    TORCH_CHECK(B2_c.is_cuda(), "B2 must be a CUDA tensor");

    TORCH_CHECK(X_c.dim() == 3, "X must be 3D");

    int B = X_c.size(0);
    int N = X_c.size(1);
    int E = X_c.size(2);
    int E_expand = W1_c.size(0);

    TORCH_CHECK(W1_c.size(1) == E, "W1 shape mismatch");
    TORCH_CHECK(W2_c.size(0) == E, "W2 shape mismatch");
    TORCH_CHECK(W2_c.size(1) == E_expand, "W2 shape mismatch");

    auto H = at::empty({B, N, E_expand}, X_c.options());
    auto O = at::empty({B, N, E}, X_c.options());

    mlp_forward_cuda(
        X_c.data_ptr<float>(),
        W1_c.data_ptr<float>(), B1_c.data_ptr<float>(),
        W2_c.data_ptr<float>(), B2_c.data_ptr<float>(),
        H.data_ptr<float>(), O.data_ptr<float>(),
        B, N, E, E_expand
    );

    return {O, H};
}