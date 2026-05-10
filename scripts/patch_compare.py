#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Ensure repo on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from inference import ViTCUDA
import vit_cuda
import timm

IMG_SIZE = 224
PATCH_SIZE = 16
PATCHES_PER_SIDE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE
PATCH_VOLUME = 3 * PATCH_SIZE * PATCH_SIZE


def load_image(path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def flatten_conv_output(conv_out):
    # conv_out shape [B, C_out, H_p, W_p]
    B, C_out, Hp, Wp = conv_out.shape
    seq = conv_out.flatten(2).transpose(1,2).contiguous()  # [B, N, C_out]
    return seq


def element_to_coord(idx):
    # idx in 0..PATCH_VOLUME-1
    c = idx // (PATCH_SIZE * PATCH_SIZE)
    rem = idx % (PATCH_SIZE * PATCH_SIZE)
    py = rem // PATCH_SIZE
    px = rem % PATCH_SIZE
    return c, py, px


def analyze_diffs(p_t_seq, p_v_seq, top_k=8):
    # p_t_seq, p_v_seq: [B, N, E]
    diff = p_t_seq - p_v_seq
    absdiff = np.abs(diff)
    # Compute per-patch maxabs and locations
    B, N, E = absdiff.shape
    summary = []
    for n in range(N):
        patch_diff = absdiff[0, n]
        maxabs = float(patch_diff.max())
        argmax = int(patch_diff.argmax())
        l2 = float(np.linalg.norm(diff[0, n].ravel()))
        summary.append((n, maxabs, argmax, l2))
    summary.sort(key=lambda x: x[1], reverse=True)
    out = []
    for r in summary[:top_k]:
        n, maxabs, argmax, l2 = r
        tval = float(p_t_seq[0, n, argmax])
        vval = float(p_v_seq[0, n, argmax])
        c, py, px = element_to_coord(argmax)
        out.append({
            'patch': n,
            'maxabs': maxabs,
            'argidx': argmax,
            'coord': {'c': int(c), 'py': int(py), 'px': int(px)},
            'tval': tval,
            'vval': vval,
            'l2': l2
        })
    return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = sys.argv[1] if len(sys.argv) > 1 else 'tests/sample.jpg'
    if not os.path.exists(path):
        print('Image not found:', path); sys.exit(1)

    print('Loading models...')
    timm_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    vit_model = ViTCUDA().to(device).eval()

    x = load_image(path, device)

    # timm patch embedding
    p_t = timm_model.patch_embed(x)
    if p_t.dim() == 4:
        p_t_seq = flatten_conv_output(p_t).detach().cpu().numpy()
    else:
        p_t_seq = p_t.detach().cpu().numpy()

    # vit_cuda patch embedding
    pw = vit_model.patch_weight
    if pw.dim() == 4:
        pw2 = pw.reshape(pw.size(0), -1).to(device)
    else:
        pw2 = pw.to(device)
    p_v = vit_cuda.patch_embed(x, pw2, vit_model.patch_bias)
    p_v_seq = p_v.cpu().numpy()

    print('Shapes: timm', p_t_seq.shape, 'vit_cuda', p_v_seq.shape)

    # Also compare raw patch vectors (unfold) vs kernel mapping to ensure same ordering
    with torch.no_grad():
        x_cpu = x.detach().cpu()
        # PyTorch unfold gives [B, C*kh*kw, L]
        p_unfold = torch.nn.functional.unfold(x_cpu, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        p_unfold_seq = p_unfold.permute(0,2,1).contiguous().numpy()  # [B, N, PATCH_VOLUME]

    # Build kernel-style patch vectors directly from image
    B = x_cpu.size(0)
    p_kernel = np.zeros_like(p_unfold_seq)
    img_np = x_cpu.numpy()
    for n in range(NUM_PATCHES):
        pr = (n // PATCHES_PER_SIDE) * PATCH_SIZE
        pc = (n % PATCHES_PER_SIDE) * PATCH_SIZE
        for c in range(3):
            for py in range(PATCH_SIZE):
                for px in range(PATCH_SIZE):
                    idx = c * (PATCH_SIZE * PATCH_SIZE) + py * PATCH_SIZE + px
                    p_kernel[0, n, idx] = img_np[0, c, pr + py, pc + px]

    # Compare unfold vs kernel mapping
    raw_diff = np.abs(p_unfold_seq - p_kernel)
    print('Raw patch vector max diff (unfold vs kernel mapping):', float(raw_diff.max()))
    diffs = analyze_diffs(p_t_seq, p_v_seq, top_k=12)
    print('\nTop differing patches:')
    for d in diffs:
        print(f"patch {d['patch']}: maxabs={d['maxabs']:.6f}, argidx={d['argidx']}, coord=(c={d['coord']['c']},py={d['coord']['py']},px={d['coord']['px']}), t={d['tval']:.6f}, v={d['vval']:.6f}, L2={d['l2']:.6f}")

    # For the worst patch, print full element-wise difference for inspection
    worst = diffs[0]
    pn = worst['patch']
    idx = worst['argidx']
    print('\nDetailed diff for worst patch, element index', idx)
    print('tval:', p_t_seq[0,pn,idx])
    print('vval:', p_v_seq[0,pn,idx])
    print('\nPatch full diff (first 32 elements):')
    np.set_printoptions(precision=6, suppress=True)
    print('t first 32:', p_t_seq[0,pn,:32])
    print('v first 32:', p_v_seq[0,pn,:32])
    print('diff first 32:', p_t_seq[0,pn,:32] - p_v_seq[0,pn,:32])

if __name__ == '__main__':
    main()
