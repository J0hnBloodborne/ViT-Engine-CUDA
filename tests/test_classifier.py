import os
import pytest
import torch

# Toggle: only run when 'classifier' present in KERNELS env
kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'classifier' not in kernels_env.split(','):
    pytest.skip('classifier tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def test_classifier_basic():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(0)

    B = 2
    SEQ = 197
    E = 768
    num_classes = 10

    X_cpu = torch.randn(B, SEQ, E, dtype=torch.float32)
    W_cpu = torch.randn(num_classes, E, dtype=torch.float32)
    bias_cpu = torch.randn(num_classes, dtype=torch.float32)

    X = X_cpu.cuda()
    W = W_cpu.cuda()
    bias = bias_cpu.cuda()

    out = vit_cuda.classifier_forward(X, W, bias)
    assert out.device.type == 'cuda'

    # Expected: use CLS token (seq index 0)
    cls = X[:, 0, :]
    expected = cls.matmul(W.t()) + bias.unsqueeze(0)

    assert expected.shape == out.shape
    assert torch.allclose(out, expected, atol=1e-4)
