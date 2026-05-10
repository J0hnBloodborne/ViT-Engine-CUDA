#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from inference import ViTCUDA
import vit_cuda


def load_image(path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def get_timm_features(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        try:
            feat = model.forward_features(x)
        except Exception:
            feat = model.forward(x)
    if feat.dim() == 3:
        feat = feat[:, 0, :]
    return feat.cpu()


def get_vitcuda_features(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        pw = model.patch_weight
        if pw.dim() == 4:
            pw2 = pw.reshape(pw.size(0), -1).to(x.device)
        else:
            pw2 = pw.to(x.device)
        # patch embedding
        xb = vit_cuda.patch_embed(x, pw2)
        B = xb.size(0)
        cls = model.cls_token
        if cls.dim() == 1:
            cls = cls.unsqueeze(0)
        cls_b = cls.expand(B, -1).contiguous().to(xb.device)
        xb = vit_cuda.pos_encoding(xb, cls_b, model.pos_embed.to(xb.device))
        for block in model.blocks:
            xb = block(xb, model.scale, model.eps)
        xb = vit_cuda.layernorm_forward(xb, model.norm_gamma.to(xb.device), model.norm_beta.to(xb.device), model.eps)
        feat = xb[:, 0, :].contiguous().cpu()
    return feat


def get_logits_timm(model, x, device):
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.cpu()


def get_logits_vitcuda(model, x, device):
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.cpu()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = 'tests/sample.jpg'
    if not os.path.exists(sample):
        print('Sample image not found:', sample); return

    x = load_image(sample)

    print('Loading timm model...')
    import timm
    timm_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)

    print('Loading ViTCUDA...')
    vit_model = ViTCUDA().to(device)

    print('Computing features...')
    t_feat = get_timm_features(timm_model, x, device)
    v_feat = get_vitcuda_features(vit_model, x, device)

    # Compare
    print('t_feat shape:', t_feat.shape)
    print('v_feat shape:', v_feat.shape)
    if t_feat.shape == v_feat.shape:
        t_norm = F.normalize(t_feat, dim=1)
        v_norm = F.normalize(v_feat, dim=1)
        cos = (t_norm * v_norm).sum(dim=1).item()
        l2 = torch.norm(t_feat - v_feat).item()
        maxabs = (t_feat - v_feat).abs().max().item()
        print(f'Cosine similarity (CLS): {cos:.6f}')
        print(f'L2 diff: {l2:.6f}, max abs diff: {maxabs:.6f}')
    else:
        print('Feature shapes differ, cannot compute direct similarity')

    print('\nTop-5 timm logits:')
    logits_t = get_logits_timm(timm_model, x, device)[0]
    probs_t = torch.nn.functional.softmax(logits_t, dim=0)
    top5_t = torch.topk(probs_t, 5)
    for p, idx in zip(top5_t.values.tolist(), top5_t.indices.tolist()):
        print(f'  {idx}: {p:.4f}')

    print('\nTop-5 ViTCUDA logits:')
    logits_v = get_logits_vitcuda(vit_model, x, device)[0]
    probs_v = torch.nn.functional.softmax(logits_v, dim=0)
    top5_v = torch.topk(probs_v, 5)
    for p, idx in zip(top5_v.values.tolist(), top5_v.indices.tolist()):
        print(f'  {idx}: {p:.4f}')

if __name__ == '__main__':
    main()
