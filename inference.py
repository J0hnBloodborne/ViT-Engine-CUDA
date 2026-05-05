
import torch

def main():
	if not torch.cuda.is_available():
		print("CUDA is not available. Please run on a CUDA-enabled machine.")
		return

	# Create a small random tensor on the GPU
	x = torch.randn(4, 4, device='cuda', dtype=torch.float32)
	x_clone = x.clone()

	try:
		import vit_cuda
	except Exception as e:
		print("Failed to import vit_cuda extension:", e)
		return

	print("Input tensor:\n", x)
	y = vit_cuda.attention_mul2(x)
	print("Output tensor:\n", y)
	# Verify result
	expected = x_clone * 2
	print("Verification (allclose):", torch.allclose(y, expected))


if __name__ == '__main__':
	main()
