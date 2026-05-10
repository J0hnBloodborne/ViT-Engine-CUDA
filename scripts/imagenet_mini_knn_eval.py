#!/usr/bin/env python3
"""
Feature extraction + leave-one-out k-NN evaluation for imagenet-mini / small ImageFolder datasets.
Saves a JSON with top1/top5 and latency numbers per backend (timm and/or vit_cuda).
"""
import argparse
import os
import json
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision import transforms, datasets

try:
    import timm
except Exception:
    timm = None

# allow importing ViTCUDA from repo
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from inference import ViTCUDA
import vit_cuda


def extract_features_vit_cuda(model, imgs):
    # model is ViTCUDA instance
    pw = model.patch_weight
    if pw.dim() == 4:
        pw2 = pw.reshape(pw.size(0), -1)
    else:
        pw2 = pw

    x = vit_cuda.patch_embed(imgs, pw2, model.patch_bias)
    B = x.size(0)
    cls = model.cls_token
    if cls.dim() == 1:
        cls = cls.unsqueeze(0)
    cls_b = cls.expand(B, -1).contiguous()
    x = vit_cuda.pos_encoding(x, cls_b, model.pos_embed)

    for block in model.blocks:
        x = block(x, model.scale, model.eps)

    x = vit_cuda.layernorm_forward(x, model.norm_gamma, model.norm_beta, model.eps)
    # CLS embedding
    feats = x[:, 0, :].detach()
    return feats


def extract_features_timm(model, imgs):
    # timm ViT has `forward_features` returning either (B, E) or (B, N, E)
    if hasattr(model, 'forward_features'):
        feats = model.forward_features(imgs)
        if feats.dim() == 3:
            feats = feats[:, 0, :]
        return feats.detach()
    else:
        # fallback: run model and try to extract penultimate layer via attribute
        out = model.forward_features(imgs)
        if out.dim() == 3:
            out = out[:, 0, :]
        return out.detach()


def run_backend(data_dir, backend, device, batch_size, limit=None, workers=4):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) == 0:
        raise SystemExit('No images found in dataset path: %s' % data_dir)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type=='cuda'))

    if backend == 'timm' and timm is None:
        raise SystemExit('timm not installed in environment')

    # instantiate
    if backend == 'timm':
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    else:
        model = ViTCUDA().to(device).eval()

    features = []
    labels = []
    latencies = []
    total_images = 0

    # warmup
    with torch.no_grad():
        it = iter(loader)
        for i in range(2):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            imgs = imgs.to(device)
            if backend == 'timm':
                _ = extract_features_timm(model, imgs)
            else:
                _ = extract_features_vit_cuda(model, imgs)

    # extraction loop
    with torch.no_grad():
        for imgs, targets in tqdm(loader):
            if limit is not None and total_images >= limit:
                break
            b = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            targets = targets

            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            if backend == 'timm':
                feats = extract_features_timm(model, imgs)
            else:
                feats = extract_features_vit_cuda(model, imgs)

            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                elapsed = 0.0

            feats = feats.cpu()
            features.append(feats)
            labels.append(targets)
            latencies.append(elapsed / float(b) if b else 0.0)

            total_images += b
            if limit is not None and total_images >= limit:
                break

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # optionally truncate to limit
    if limit is not None and features.size(0) > limit:
        features = features[:limit]
        labels = labels[:limit]

    # normalize features and compute similarity
    features = F.normalize(features, p=2, dim=1)
    sims = features @ features.t()
    n = sims.size(0)
    # exclude self
    sims.fill_diagonal_(-9e9)

    k = 5
    topk = torch.topk(sims, k=k, dim=1).indices

    top1 = (labels[topk[:, 0]] == labels).sum().item()
    top5 = (labels[topk] == labels.unsqueeze(1)).any(dim=1).sum().item()

    mean_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0
    median_latency = float(sorted(latencies)[len(latencies)//2]) if latencies else 0.0
    images_per_second = (features.size(0) / (sum(latencies)/1000.0)) if sum(latencies) > 0 else 0.0

    results = {
        'backend': backend,
        'device': device.type,
        'total_images': int(features.size(0)),
        'top1': int(top1),
        'top5': int(top5),
        'top1_acc': float(top1) / float(features.size(0)) if features.size(0) else 0.0,
        'top5_acc': float(top5) / float(features.size(0)) if features.size(0) else 0.0,
        'mean_latency_ms_per_image': mean_latency,
        'median_latency_ms_per_image': median_latency,
        'images_per_second': images_per_second,
        'latencies_ms_samples': latencies[:100],
    }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--backend', choices=['timm', 'vit_cuda', 'both'], default='both')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save', type=str, default='results/imagenet_mini_knn_results.json')
    args = p.parse_args()

    backends = ['timm', 'vit_cuda'] if args.backend == 'both' else [args.backend]
    out = {}
    for b in backends:
        print('Running backend:', b)
        res = run_backend(args.data, b, args.device, args.batch_size, limit=args.limit, workers=args.workers)
        out[b] = res
        print(json.dumps(res, indent=2))

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved results to', args.save)


if __name__ == '__main__':
    main()
