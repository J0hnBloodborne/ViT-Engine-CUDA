import torch
import vit_cuda
import math

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

M = B * N
X_mat_gpu = X.reshape(M, E)
def gelu_approx(x):
    SQRT_2_OVER_PI = 0.7978845608
    COEF = 0.044715
    z = SQRT_2_OVER_PI * x * (1.0 + COEF * x * x)
    z = torch.clamp(z, -20.0, 20.0)
    return 0.5 * x * (1.0 + torch.tanh(z))

H_expected_gpu = X_mat_gpu.matmul(W1.t()) + B1.unsqueeze(0)
H_expected_gpu = gelu_approx(H_expected_gpu)
O_expected_gpu = H_expected_gpu.matmul(W2.t()) + B2.unsqueeze(0)

O_expected_gpu = O_expected_gpu.view(B, N, E)
O_cuda = O_cuda.contiguous()

diff = (O_cuda - O_expected_gpu).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
print(f"max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}")
inds = (diff > 1e-3).nonzero(as_tuple=False)
print(f"num>1e-3 = {inds.shape[0]}")
if inds.shape[0] > 0:
    i0 = inds[0].tolist()
    print("example at", i0, "cuda_val", O_cuda[tuple(i0)].item(), "expected", O_expected_gpu[tuple(i0)].item(), "diff", diff[tuple(i0)].item())
rel = diff / (O_expected_gpu.abs() + 1e-8)
print(f"max_rel_diff={rel.max().item():.6g}")
