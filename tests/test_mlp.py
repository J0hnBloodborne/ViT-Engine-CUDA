import os
import pytest
import torch

# Toggle: only run when 'mlp' present in KERNELS env
kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'mlp' not in kernels_env.split(','):
    pytest.skip('mlp tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def gelu_approx(x):
    # Match the kernel's approximation with clamping used there
    SQRT_2_OVER_PI = 0.7978845608
    COEF = 0.044715
    z = SQRT_2_OVER_PI * x * (1.0 + COEF * x * x)
    z = torch.clamp(z, -20.0, 20.0)
    return 0.5 * x * (1.0 + torch.tanh(z))


def test_mlp_basic():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(0)

    B = 1
    N = 3
    E = 768
    E_expand = 3072

    X_cpu = torch.randn(B, N, E, dtype=torch.float32)
    W1_cpu = torch.randn(E_expand, E, dtype=torch.float32)
    B1_cpu = torch.randn(E_expand, dtype=torch.float32)
    W2_cpu = torch.randn(E, E_expand, dtype=torch.float32)
    B2_cpu = torch.randn(E, dtype=torch.float32)

    X = X_cpu.cuda()
    W1 = W1_cpu.cuda()
    B1 = B1_cpu.cuda()
    W2 = W2_cpu.cuda()
    B2 = B2_cpu.cuda()

    O_cuda, H_cuda = vit_cuda.mlp_forward(X, W1, B1, W2, B2)
    assert O_cuda.device.type == 'cuda'
    assert H_cuda.device.type == 'cuda'

    O_cpu = O_cuda.cpu()
    H_cpu = H_cuda.cpu()

    # Compute expected on GPU so both sides use the same FP accumulation
    M = B * N
    X_mat_gpu = X.reshape(M, E)
    H_expected_gpu = X_mat_gpu.matmul(W1.t()) + B1.unsqueeze(0)
    H_expected_gpu = gelu_approx(H_expected_gpu)
    O_expected_gpu = H_expected_gpu.matmul(W2.t()) + B2.unsqueeze(0)

    H_expected_gpu = H_expected_gpu.view(B, N, E_expand)
    O_expected_gpu = O_expected_gpu.view(B, N, E)

    assert O_expected_gpu.shape == O_cuda.shape
    assert H_expected_gpu.shape == H_cuda.shape

    assert torch.allclose(H_cuda, H_expected_gpu, atol=1e-4)
    # Allow small relative differences for the final output due to different accumulation/order in kernels
    rel = (O_cuda - O_expected_gpu).abs() / (O_expected_gpu.abs() + 1e-8)
    max_rel = rel.max().item()
    assert max_rel < 3e-3, f"max relative diff too large: {max_rel}"
