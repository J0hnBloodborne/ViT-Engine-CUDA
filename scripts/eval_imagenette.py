#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
import urllib.request
import tarfile
import random

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

import numpy as np

# Ensure repo root is on sys.path so we can import `inference` when running from `scripts/`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import local ViT wrapper
from inference import ViTCUDA
import vit_cuda

IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'


def download_imagenette(dest_root):
    if os.path.isdir(dest_root):
        print(dest_root, 'already exists; skipping download')
        return dest_root
    tgz_path = dest_root + '.tgz'
    os.makedirs(os.path.dirname(dest_root), exist_ok=True)
    print('Downloading Imagenette (this may take a minute)...')
    urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
    print('Extracting...')
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(dest_root))
    os.remove(tgz_path)
    return dest_root


def vitcuda_features(model, img_batch):
    # Recreate forward up to final layernorm, return CLS embedding (B, E)
    pw = model.patch_weight
    if pw.dim() == 4:
        pw2 = pw.reshape(pw.size(0), -1)
    else:
        pw2 = pw

    x = vit_cuda.patch_embed(img_batch, pw2, model.patch_bias)
    B = x.size(0)
    cls = model.cls_token
    if cls.dim() == 1:
        cls = cls.unsqueeze(0)
    cls_b = cls.expand(B, -1).contiguous().to(x.device)

    x = vit_cuda.pos_encoding(x, cls_b, model.pos_embed)
    for block in model.blocks:
        x = block(x, model.scale, model.eps)
    x = vit_cuda.layernorm_forward(x, model.norm_gamma, model.norm_beta, model.eps)
    feat = x[:, 0, :].contiguous()
    return feat


def extract_features(backend, dataloader, model, device, limit=None):
    feats = []
    labels = []
    latencies = []
    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            if limit is not None and len(labels) >= limit:
                break
            b = imgs.size(0)
            imgs = imgs.to(device)
            targets = targets.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            if backend == 'timm':
                try:
                    feat = model.forward_features(imgs)
                except Exception:
                    feat = model.forward(imgs)
            elif backend == 'vit_cuda':
                feat = vitcuda_features(model, imgs)
            else:
                raise ValueError('Unknown backend: %s' % backend)

            # If model returned per-token features (B, N, E), take CLS token
            if hasattr(feat, 'dim') and feat.dim() == 3:
                feat = feat[:, 0, :]

            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)  # ms per batch
                per_image = elapsed / float(b)
            else:
                per_image = 0.0

            feats.append(feat.cpu())
            labels.append(targets.cpu())
            latencies.extend([per_image] * b)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    if limit is not None:
        feats = feats[:limit]
        labels = labels[:limit]
        latencies = latencies[:len(feats)]
    return feats, labels, latencies


def knn_classify(train_feats, train_labels, test_feats, test_labels, k=5, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    train = F.normalize(train_feats, p=2, dim=1).to(device)
    test = F.normalize(test_feats, p=2, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    sims = test @ train.t()
    topk = sims.topk(k, dim=1, largest=True, sorted=True)
    topk_idx = topk.indices
    topk_labels = train_labels[topk_idx]

    top1 = (topk_labels[:, 0] == test_labels).sum().item()
    topk_any = (topk_labels == test_labels.unsqueeze(1)).any(dim=1).sum().item()
    total = test.size(0)
    return {'top1': int(top1), 'topk_any': int(topk_any), 'total': int(total), 'k': int(k)}


def plot_pca(feats, labels, out_path, n_samples=500):
    import matplotlib.pyplot as plt
    X = feats.numpy()
    y = labels.numpy()
    n = X.shape[0]
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    Xs = X[idx]
    ys = y[idx]
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt.T[:, :2]
    plt.figure(figsize=(8,6))
    num_classes = len(np.unique(ys))
    cmap = plt.cm.get_cmap('tab10', num_classes)
    for c in range(num_classes):
        sel = (ys == c)
        plt.scatter(coords[sel,0], coords[sel,1], s=8, color=cmap(c), label=str(c), alpha=0.7)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('PCA of embeddings')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/imagenette2-320', help='download/extract here if missing')
    parser.add_argument('--backend', choices=['timm','vit_cuda','both'], default='both')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--limit', type=int, default=200, help='limit per split (train/val)')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--save', default='results/imagenette_results.json')
    parser.add_argument('--plot', default='results/imagenette_pca.png')
    args = parser.parse_args()

    # Ensure matplotlib available
    try:
        import matplotlib
    except Exception:
        print('Installing matplotlib...')
        os.system(f"{sys.executable} -m pip install matplotlib")

    if args.backend in ('both','timm'):
        try:
            import timm
        except Exception:
            print('Please install timm in the environment (pip install timm)'); return

    # Download if needed
    dataset_root = args.data
    if not os.path.isdir(dataset_root):
        download_imagenette(dataset_root)

    # Determine train/val paths
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        candidate = os.path.join(os.path.dirname(dataset_root), 'imagenette2-320')
        train_dir = os.path.join(candidate, 'train')
        val_dir = os.path.join(candidate, 'val')

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise SystemExit('Could not find Imagenette train/val after extraction')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    limit = args.limit
    if limit is not None and limit < len(train_ds):
        idx = torch.randperm(len(train_ds))[:limit].tolist()
        train_ds = Subset(train_ds, idx)
    if limit is not None and limit < len(val_ds):
        idx = torch.randperm(len(val_ds))[:limit].tolist()
        val_ds = Subset(val_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(args.device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(args.device=='cuda'))

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    results = {}

    if args.backend in ('both','timm'):
        print('Preparing timm model...')
        import timm
        timm_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
        print('Extracting timm features (train)...')
        ttrain_feats, ttrain_labels, ttrain_lat = extract_features('timm', train_loader, timm_model, device, limit=None)
        print('Extracting timm features (val)...')
        tval_feats, tval_labels, tval_lat = extract_features('timm', val_loader, timm_model, device, limit=None)
        knn = knn_classify(ttrain_feats, ttrain_labels, tval_feats, tval_labels, k=args.k, device=args.device)
        results['timm'] = {
            'knn': knn,
            'mean_latency_ms_per_image': float(np.mean(ttrain_lat + tval_lat)) if (ttrain_lat + tval_lat) else 0.0
        }

    if args.backend in ('both','vit_cuda'):
        print('Preparing ViTCUDA model...')
        vit_model = ViTCUDA().to(device).eval()
        print('Extracting vit_cuda features (train)...')
        vtrain_feats, vtrain_labels, vtrain_lat = extract_features('vit_cuda', train_loader, vit_model, device, limit=None)
        print('Extracting vit_cuda features (val)...')
        vval_feats, vval_labels, vval_lat = extract_features('vit_cuda', val_loader, vit_model, device, limit=None)
        knn = knn_classify(vtrain_feats, vtrain_labels, vval_feats, vval_labels, k=args.k, device=args.device)
        results['vit_cuda'] = {
            'knn': knn,
            'mean_latency_ms_per_image': float(np.mean(vtrain_lat + vval_lat)) if (vtrain_lat + vval_lat) else 0.0
        }

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', args.save)

    # Plot PCA of val embeddings (prefer vit_cuda if available)
    if 'vit_cuda' in results:
        plot_feats = vval_feats
        plot_labels = vval_labels
    elif 'timm' in results:
        plot_feats = tval_feats
        plot_labels = tval_labels
    else:
        plot_feats = None

    if plot_feats is not None:
        plot_pca(plot_feats, plot_labels, args.plot, n_samples=500)
        print('Saved PCA plot to', args.plot)


if __name__ == '__main__':
    main()
