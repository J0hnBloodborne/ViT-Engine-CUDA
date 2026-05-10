import torch
import vit_cuda

def test_mlp_batch():
    B = 16
    N = 197
    E = 768
    E_expand = 3072
    
    x = torch.randn(B, N, E).cuda()
    w1 = torch.randn(E_expand, E).cuda()
    b1 = torch.randn(E_expand).cuda()
    w2 = torch.randn(E, E_expand).cuda()
    b2 = torch.randn(E).cuda()
    
    out, hidden = vit_cuda.mlp_forward(x, w1, b1, w2, b2)
    
    print(f"Output shape: {out.shape}")
    
    diff = (out[0] - out[1]).abs().max().item()
    print(f"Max diff between batch 0 and 1: {diff}")
    
    if diff == 0:
        print("BUG REPRODUCED: Batch images are identical in MLP!")
    else:
        print("MLP batch images are different.")

if __name__ == "__main__":
    test_mlp_batch()
