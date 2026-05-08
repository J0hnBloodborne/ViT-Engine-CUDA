#include <torch/extension.h>

// Forward declaration of the CUDA launcher implemented in pos_encoding.cu
void launch_pos_encoding(float* patches, float* cls_token, float* pos_embed, float* out, int batch_size);

at::Tensor pos_encoding(at::Tensor patches, at::Tensor cls_token, at::Tensor pos_embeddings) { // Defines the C++ function accepting PyTorch tensors
    // Pytorch checks for input validity
	TORCH_CHECK(patches.is_cuda(), "patches must be a CUDA tensor");
	TORCH_CHECK(cls_token.is_cuda(), "cls_token must be a CUDA tensor");
	TORCH_CHECK(pos_embeddings.is_cuda(), "pos_embeddings must be a CUDA tensor");
	TORCH_CHECK(patches.scalar_type() == at::kFloat, "patches must be float32");
	TORCH_CHECK(cls_token.scalar_type() == at::kFloat, "cls_token must be float32");
	TORCH_CHECK(pos_embeddings.scalar_type() == at::kFloat, "pos_embeddings must be float32");

	at::Tensor p_c = patches.contiguous(); // Ensures the patches tensor is contiguous in memory
	at::Tensor c_c = cls_token.contiguous(); // Ensures the cls_token tensor is contiguous in memory
	at::Tensor pos_c = pos_embeddings.contiguous(); // Ensures the pos_embeddings tensor is contiguous in memory

	int64_t batch_size = p_c.size(0); // Extracts the batch size
	int64_t num_patches = p_c.size(1); // Extracts the number of patches
	int64_t embed_dim = p_c.size(2); // Extracts the embedding dimension

    // More checks to ensure cls_token and pos_embeddings have the expected shapes
	TORCH_CHECK(c_c.dim() == 2, "cls_token must be [B,embed_dim]"); 
	TORCH_CHECK(c_c.size(0) == batch_size, "cls_token batch size mismatch");
	TORCH_CHECK(c_c.size(1) == embed_dim, "cls_token embed dim mismatch");

	TORCH_CHECK(pos_c.dim() == 2, "pos_embeddings must be [SEQ_LEN,embed_dim]");
	TORCH_CHECK(pos_c.size(1) == embed_dim, "pos_embeddings embed dim mismatch");
	TORCH_CHECK(pos_c.size(0) == (num_patches + 1), "pos_embeddings length must equal num_patches + 1 (including CLS token)");

	at::Tensor out = at::zeros({batch_size, num_patches + 1, embed_dim}, p_c.options()); // Creates an output tensor initialized to zeros with shape [B, SEQ_LEN, embed_dim]

	launch_pos_encoding(p_c.data_ptr<float>(), c_c.data_ptr<float>(), pos_c.data_ptr<float>(), out.data_ptr<float>(), (int)batch_size); // Calls the CUDA launcher function

	return out;
}

