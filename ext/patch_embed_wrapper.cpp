#include <torch/extension.h>

// Forward declaration of the CUDA launcher implemented in patch_embed.cu
void launch_patch_embed(float* img, float* weights, float* out, int B);

at::Tensor patch_embed(at::Tensor img, at::Tensor weights, c10::optional<at::Tensor> bias_opt = c10::nullopt) { // Defines the C++ function accepting PyTorch tensors
    // Pytorch checks for input validity
    TORCH_CHECK(img.is_cuda(), "img must be a CUDA tensor"); 
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(img.scalar_type() == at::kFloat, "img must be float32");
    TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be float32");
    TORCH_CHECK(img.dim() == 4, "img must be [B,C,H,W]");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
    }

    at::Tensor img_c = img.contiguous(); // Ensures the input tensor is contiguous in memory
    at::Tensor w_c = weights.contiguous(); // Ensures the weights tensor is contiguous in memory

    int64_t B = img_c.size(0); // Extracts the batch size
    int64_t H = img_c.size(2); // Extracts the height of the input image
    int64_t W = img_c.size(3); // Extracts the width of the input image
    int64_t patch_size = 16; 
    int64_t num_patches = (H / patch_size) * (W / patch_size);
    int64_t embed_dim = w_c.size(0); // Extracts the embedding dimension from the weights tensor

    at::Tensor out = at::zeros({B, num_patches, embed_dim}, img_c.options()); // Creates an output tensor initialized to zeros

    launch_patch_embed(img_c.data_ptr<float>(), w_c.data_ptr<float>(), out.data_ptr<float>(), (int)B); // Calls the CUDA launcher function with raw pointers to the data and batch size

    if (bias_opt.has_value()) {
        out = out + bias_opt.value().contiguous().view({1, 1, -1});
    }

    return out;
}
