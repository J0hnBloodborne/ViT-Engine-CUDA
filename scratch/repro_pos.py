import torch
import vit_cuda

def test_pos_encoding_batch():
    B = 16
    N = 196
    E = 768
    
    patches = torch.randn(B, N, E).cuda()
    cls_token = torch.randn(B, E).cuda()
    pos_embed = torch.randn(N + 1, E).cuda()
    
    out = vit_cuda.pos_encoding(patches, cls_token, pos_embed)
    
    print(f"Output shape: {out.shape}")
    
    diff = (out[0] - out[1]).abs().max().item()
    print(f"Max diff between batch 0 and 1: {diff}")
    
    if diff == 0:
        print("BUG REPRODUCED: Batch images are identical in pos_encoding!")
    else:
        print("pos_encoding batch images are different.")

if __name__ == "__main__":
    test_pos_encoding_batch()
