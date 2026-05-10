import torch
import vit_cuda

def test_layernorm_batch():
    B = 16
    N = 197
    E = 768
    
    x = torch.randn(B, N, E).cuda()
    gamma = torch.randn(E).cuda()
    beta = torch.randn(E).cuda()
    
    out = vit_cuda.layernorm_forward(x, gamma, beta, 1e-6)
    
    print(f"Output shape: {out.shape}")
    
    diff = (out[0] - out[1]).abs().max().item()
    print(f"Max diff between batch 0 and 1: {diff}")
    
    if diff == 0:
        print("BUG REPRODUCED: Batch images are identical in layernorm!")
    else:
        print("layernorm batch images are different.")

if __name__ == "__main__":
    test_layernorm_batch()
