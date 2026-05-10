import os
import pytest
import torch

# Toggle: only run when 'layernorm' present in KERNELS env
kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'layernorm' not in kernels_env.split(','):
    pytest.skip('layernorm tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def test_layernorm_basic():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    # Must match kernel compile-time EMBED dim
    B = 1
    N = 4
    E = 768
    eps = 1e-5

    torch.manual_seed(0)
    X_cpu = torch.randn(B, N, E, dtype=torch.float32)
    gamma_cpu = torch.randn(E, dtype=torch.float32)
    beta_cpu = torch.randn(E, dtype=torch.float32)

    X = X_cpu.cuda()
    gamma = gamma_cpu.cuda()
    beta = beta_cpu.cuda()

    out = vit_cuda.layernorm_forward(X, gamma, beta, float(eps))
    assert out.device.type == 'cuda'

    out_cpu = out.cpu()

    expected = torch.nn.functional.layer_norm(X_cpu, (E,), weight=gamma_cpu, bias=beta_cpu, eps=eps)

    assert expected.shape == out_cpu.shape
    assert torch.allclose(out_cpu, expected, atol=1e-4)
