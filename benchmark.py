import time
import torch
import timm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from inference import ViTCUDA


def measure_latency(model, inp, iterations=100, warmup=10):
    # ensure model on eval and CUDA
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)

    torch.cuda.synchronize()
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        starter.record()
        with torch.no_grad():
            _ = model(inp)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    # times in milliseconds
    avg_ms = sum(times) / len(times)
    return avg_ms


def main():
    device = 'cuda'
    batch = 32
    inp = torch.randn(batch, 3, 224, 224, device=device, dtype=torch.float32)

    # timm reference model (PyTorch)
    ref = timm.create_model('vit_base_patch16_224', pretrained=False).to(device).eval()

    # custom compiled model
    vit = ViTCUDA().to(device).eval()

    print('Warming and measuring...')
    ref_ms = measure_latency(ref, inp, iterations=100, warmup=10)
    vit_ms = measure_latency(vit, inp, iterations=100, warmup=10)

    print(f'Reference timm ViT average latency: {ref_ms:.3f} ms')
    print(f'ViTCUDA average latency: {vit_ms:.3f} ms')
    if vit_ms > 0:
        print(f'Speedup (timm / vit_cuda): {ref_ms / vit_ms:.2f}x')


if __name__ == '__main__':
    main()
