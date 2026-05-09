import torch
import torch.nn.functional as F
import vit_cuda
import math

def manual_attention(q, k, v, scale):
    # Standard attention formula for reference
    # q, k, v shape: [B, H, N, D]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

def test_flash_attn_correctness():
    B, H, N, D = 2, 4, 64, 64
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    # Reference output
    expected = manual_attention(q, k, v, scale)

    # Flash Attention 2 output
    actual = vit_cuda.flash_attn_2(q, k, v, scale)

    # Check difference
    diff = (expected - actual).abs().max().item()
    print(f"Max absolute difference: {diff}")

    if diff < 1e-4:
        print("Test PASSED!")
    else:
        print("Test FAILED!")

if __name__ == "__main__":
    test_flash_attn_correctness()
