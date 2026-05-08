import os
import pytest
import torch

# Toggle: only run when 'patch_embed' present in KERNELS env (default enabled)
kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'patch_embed' not in kernels_env.split(','):
    pytest.skip('patch_embed tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def test_patch_embed_basic():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    # constants must match the kernel
    B = 1
    C = 3
    IMG_SIZE = 224
    PATCH_SIZE = 16
    PATCH_AREA = PATCH_SIZE * PATCH_SIZE
    PATCH_VOLUME = C * PATCH_AREA
    EMBED_DIM = 768
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE)

    # deterministic input
    torch.manual_seed(0)
    img_cpu = torch.randn(B, C, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    weights_cpu = torch.randn(EMBED_DIM, PATCH_VOLUME, dtype=torch.float32)

    img = img_cpu.cuda()
    weights = weights_cpu.cuda()

    out = vit_cuda.patch_embed(img, weights)
    assert out.device.type == 'cuda'

    out_cpu = out.cpu()

    # build expected output on CPU by extracting patches and matrix-multiplying
    patches = []
    for p in range(NUM_PATCHES):
        row = (p // (IMG_SIZE // PATCH_SIZE)) * PATCH_SIZE
        col = (p % (IMG_SIZE // PATCH_SIZE)) * PATCH_SIZE
        patch = img_cpu[0, :, row:row+PATCH_SIZE, col:col+PATCH_SIZE].reshape(-1)
        patches.append(patch)
    patches = torch.stack(patches, dim=0)  # [num_patches, patch_volume]

    expected = patches.matmul(weights_cpu.t()).unsqueeze(0)  # [1, num_patches, embed_dim]

    assert expected.shape == out_cpu.shape
    assert torch.allclose(out_cpu, expected, atol=1e-4)
