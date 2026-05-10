import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import vit_cuda
import math

def manual_attention(q, k, v, scale):
    # Standard attention formula for reference
    # q, k, v shape: [B, N, 768]
    # Reshape to [B, N, 12, 64] and then transpose to [B, 12, N, 64]
    B, N, _ = q.shape
    q = q.view(B, N, 12, 64).transpose(1, 2)
    k = k.view(B, N, 12, 64).transpose(1, 2)
    v = v.view(B, N, 12, 64).transpose(1, 2)

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    # Transpose back to [B, N, 12, 64] and flatten to [B, N, 768]
    return output.transpose(1, 2).contiguous().view(B, N, 768)

def test_flash_attn_correctness():
    # ViT typical parameters
    B, N, E = 2, 197, 768
    D = E // 12
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, N, E, device='cuda', dtype=torch.float32)
    k = torch.randn(B, N, E, device='cuda', dtype=torch.float32)
    v = torch.randn(B, N, E, device='cuda', dtype=torch.float32)

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
