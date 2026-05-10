#include <torch/extension.h>

// Forward declaration of CUDA launcher implemented in layernorm.cu
void launch_layernorm(const float* X, const float* gamma, const float* beta, float* Y, int B, int N, float eps);

at::Tensor layernorm_forward(at::Tensor X, at::Tensor gamma, at::Tensor beta, float eps) {
	TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor"); 
	TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
	TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");

	TORCH_CHECK(X.scalar_type() == at::kFloat, "X must be float32");
	TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
	TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");

	TORCH_CHECK(X.dim() == 3, "X must be [B, N, E]");

	auto Xc = X.contiguous();
	auto gc = gamma.contiguous();
	auto bc = beta.contiguous();

	int B = (int)Xc.size(0);
	int N = (int)Xc.size(1);
	int E = (int)Xc.size(2);

	TORCH_CHECK((int)gc.numel() == E, "gamma size mismatch");
	TORCH_CHECK((int)bc.numel() == E, "beta size mismatch");

	auto Y = at::empty_like(Xc);

	launch_layernorm(Xc.data_ptr<float>(), gc.data_ptr<float>(), bc.data_ptr<float>(), Y.data_ptr<float>(), B, N, eps);

	return Y;
}
