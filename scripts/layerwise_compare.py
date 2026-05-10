#!/usr/bin/env python3
import os
import sys
import argparse
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
import timm


def preprocess_image(path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def as_np(t):
    return t.detach().cpu().numpy()


def compare_np(a, b):
    res = {}
    res['shape_a'] = tuple(a.shape)
    res['shape_b'] = tuple(b.shape)
    if a.shape != b.shape:
        res['mismatch'] = True
        return res
    diff = a - b
    ad = np.abs(diff)
    res['mismatch'] = False
    res['l2'] = float(np.linalg.norm(diff.ravel()))
    res['maxabs'] = float(ad.max())
    res['meanabs'] = float(ad.mean())
    # cosine on flattened vectors
    af = a.ravel()
    bf = b.ravel()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    res['cosine'] = float(np.dot(af, bf) / denom)
    return res


def fmt(m):
    if m.get('mismatch'):
        return f"shapes differ: {m['shape_a']} vs {m['shape_b']}"
    return f"L2={m['l2']:.6f}, maxabs={m['maxabs']:.6f}, meanabs={m['meanabs']:.6f}, cos={m['cosine']:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='tests/sample.jpg')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    img_path = args.image
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        sys.exit(1)

    print('Loading models...')
    timm_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    vit_model = ViTCUDA().to(device).eval()

    x = preprocess_image(img_path, device)

    B = x.size(0)

    print('\n== Patch embedding ==')
    # timm patch
    p_t = timm_model.patch_embed(x)
    if p_t.dim() == 4:
        Bp, C, H, W = p_t.shape
        p_t_seq = p_t.flatten(2).transpose(1, 2).contiguous()
    else:
        p_t_seq = p_t

    # vit_cuda patch
    pw = vit_model.patch_weight
    if pw.dim() == 4:
        pw2 = pw.reshape(pw.size(0), -1).to(device)
    else:
        pw2 = pw.to(device)
    p_v_seq = vit_cuda.patch_embed(x, pw2)

    m_patch = compare_np(as_np(p_t_seq), as_np(p_v_seq))
    print('patch compare:', fmt(m_patch))

    print('\n== Positional encoding (with cls) ==')
    # timm: build sequence with cls
    cls_t = timm_model.cls_token
    if cls_t.dim() == 3:
        cls_t_exp = cls_t.expand(B, -1, -1).to(device)
    elif cls_t.dim() == 2:
        cls_t_exp = cls_t.unsqueeze(1).expand(B,1,-1).to(device)
    elif cls_t.dim() == 1:
        cls_t_exp = cls_t.unsqueeze(0).unsqueeze(1).expand(B,1,-1).to(device)
    else:
        raise RuntimeError('Unexpected cls_token dim')

    pos_t = timm_model.pos_embed
    if pos_t.dim() == 3:
        pos_t_b = pos_t.to(device)
    elif pos_t.dim() == 2:
        pos_t_b = pos_t.unsqueeze(0).to(device)
    else:
        raise RuntimeError('Unexpected pos_embed dim')

    x_t = torch.cat((cls_t_exp, p_t_seq), dim=1)
    x_t_pos = x_t + pos_t_b

    # vit pos encoding expects cls_b shape [B, E]
    cls_v = vit_model.cls_token
    if cls_v.dim() == 1:
        cls_v_b = cls_v.unsqueeze(0).expand(B, -1).contiguous().to(device)
    elif cls_v.dim() == 2:
        cls_v_b = cls_v.expand(B, -1).contiguous().to(device)
    elif cls_v.dim() == 3:
        # e.g. [1,1,E]
        cls_v_b = cls_v.squeeze(1).expand(B, -1).contiguous().to(device)
    else:
        raise RuntimeError('Unexpected cls_token dim for vit_model')

    pos_v = vit_model.pos_embed.to(device)
    x_v_pos = vit_cuda.pos_encoding(p_v_seq, cls_v_b, pos_v)

    m_pos = compare_np(as_np(x_t_pos), as_np(x_v_pos))
    print('pos compare:', fmt(m_pos))

    # iterate blocks
    x_t_cur = x_t_pos
    x_v_cur = x_v_pos
    num_blocks = len(timm_model.blocks)
    print('\n== Per-block comparison ==')
    for i in range(num_blocks):
        block_t = timm_model.blocks[i]
        block_v = vit_model.blocks[i]
        x_t_next = block_t(x_t_cur)
        # vit block signature: block(x, scale, eps)
        x_v_next = block_v(x_v_cur, vit_model.scale, vit_model.eps)

        m_block = compare_np(as_np(x_t_next), as_np(x_v_next))
        print(f'block {i}:', fmt(m_block))

        x_t_cur = x_t_next
        x_v_cur = x_v_next

    print('\n== Final layernorm compare ==')
    x_t_norm = timm_model.norm(x_t_cur)
    x_v_norm = vit_cuda.layernorm_forward(x_v_cur, vit_model.norm_gamma.to(x_v_cur.device), vit_model.norm_beta.to(x_v_cur.device), vit_model.eps)
    m_norm = compare_np(as_np(x_t_norm), as_np(x_v_norm))
    print('final norm:', fmt(m_norm))

    # return nonzero exit code if large mismatch found
    thresholds = {'maxabs': 1e-3, 'l2': 1e-2}
    large = False
    for label, val in [('patch', m_patch), ('pos', m_pos)]:
        if not val.get('mismatch') and (val['maxabs'] > thresholds['maxabs'] or val['l2'] > thresholds['l2']):
            large = True
    for i in range(num_blocks):
        pass

    if large:
        print('\nNOTE: Large differences detected. Suggest inspecting `patch_embed` and `pos_encoding` kernels first.')
        sys.exit(2)
    else:
        print('\nAll layers similar within thresholds')


if __name__ == '__main__':
    main()
