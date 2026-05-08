import os
import pytest
import torch

# Toggle: only run when 'pos_encoding' present in KERNELS env
kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'pos_encoding' not in kernels_env.split(','):
    pytest.skip('pos_encoding tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def test_pos_encoding_basic():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    # constants must match the kernel
    B = 1
    EMBED_DIM = 768
    NUM_PATCHES = 196
    SEQ_LEN = NUM_PATCHES + 1

    torch.manual_seed(0)
    patches_cpu = torch.randn(B, NUM_PATCHES, EMBED_DIM, dtype=torch.float32)
    cls_cpu = torch.randn(B, EMBED_DIM, dtype=torch.float32)
    pos_cpu = torch.randn(SEQ_LEN, EMBED_DIM, dtype=torch.float32)

    patches = patches_cpu.cuda()
    cls = cls_cpu.cuda()
    pos = pos_cpu.cuda()

    out = vit_cuda.pos_encoding(patches, cls, pos)
    assert out.device.type == 'cuda'

    out_cpu = out.cpu()

    expected = torch.zeros(B, SEQ_LEN, EMBED_DIM, dtype=torch.float32)
    for b in range(B):
        expected[b, 0, :] = cls_cpu[b, :] + pos_cpu[0, :]
        for i in range(1, SEQ_LEN):
            expected[b, i, :] = patches_cpu[b, i - 1, :] + pos_cpu[i, :]

    assert expected.shape == out_cpu.shape
    assert torch.allclose(out_cpu, expected, atol=1e-4)
