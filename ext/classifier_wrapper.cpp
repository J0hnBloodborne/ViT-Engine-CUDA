#include <torch/extension.h>

// Forward declaration of the CUDA launcher implemented in classifier.cu
void launch_classifier(const float* X, const float* W, const float* bias, float* Y, int B, int num_classes);

at::Tensor classifier_forward(at::Tensor X, at::Tensor W, at::Tensor bias) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(X.scalar_type() == at::kFloat, "X must be float32");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(X.dim() == 3, "X must be [B, SEQ_LEN, E]");
    TORCH_CHECK(W.dim() == 2, "W must be [num_classes, E]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [num_classes]");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    auto bc = bias.contiguous();

    int B = (int)Xc.size(0);
    int SEQ = (int)Xc.size(1);
    int E = (int)Xc.size(2);
    int num_classes = (int)Wc.size(0);

    TORCH_CHECK((int)Wc.size(1) == E, "W second dim must match embed dim");
    TORCH_CHECK((int)bc.numel() == num_classes, "bias size mismatch");

    auto Y = at::empty({B, num_classes}, X.options());

    launch_classifier(Xc.data_ptr<float>(), Wc.data_ptr<float>(), bc.data_ptr<float>(), Y.data_ptr<float>(), B, num_classes);

    return Y;
}
