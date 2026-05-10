import torch
import vit_cuda
import numpy as np

def test_patch_embed_batch():
    B = 16
    C = 3
    H = 224
    W = 224
    E = 768
    
    img = torch.randn(B, C, H, W).cuda()
    weights = torch.randn(E, C * 16 * 16).cuda()
    bias = torch.randn(E).cuda()
    
    out = vit_cuda.patch_embed(img, weights, bias)
    
    print(f"Output shape: {out.shape}")
    
    # Check if image 0 and image 1 are different
    diff = (out[0] - out[1]).abs().max().item()
    print(f"Max diff between batch 0 and 1: {diff}")
    
    if diff == 0:
        print("BUG REPRODUCED: Batch images are identical!")
    else:
        print("Batch images are different. Bug not reproduced or fixed.")

if __name__ == "__main__":
    test_patch_embed_batch()
