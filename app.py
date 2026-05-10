import os
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from inference import ViTCUDA, get_imagenet_labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_labels():
    try:
        return get_imagenet_labels()
    except Exception:
        return [f"class_{i}" for i in range(1000)]


model = None
labels = None


def init_model():
    global model
    if model is None:
        model = ViTCUDA().cuda().eval()


def predict_image(img):
    init_model()
    if img is None:
        return {}

    img_t = preprocess(img).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top5_prob, top5_idx = torch.topk(probs, 5)
    results = []
    for p, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
        lbl = labels[idx] if idx < len(labels) else str(idx)
        results.append((lbl, float(p)))

    return {r[0]: r[1] for r in results}


def main():
    global labels
    labels = load_labels()

    # Use a single upload input to maximize compatibility across Gradio versions
    try:
        input_component = gr.Image(type="pil", label="Upload")
    except TypeError:
        # Fallback for older Gradio
        input_component = gr.inputs.Image(type="pil", label="Upload")

    iface = gr.Interface(
        fn=predict_image,
        inputs=[input_component],
        outputs=gr.Label(num_top_classes=5),
        title="ViTCUDA ImageNet Demo",
        description="Upload an image. Runs on GPU with compiled vit_cuda backend.",
        flagging_mode="never",
    )
    iface.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
import argparse
import math
import urllib.request
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# Import your compiled C++ extension
import vit_cuda

class ViTBlockCUDA(nn.Module):
    def __init__(self, state, idx):
        super().__init__()
        prefix = f'blocks.{idx}.'
        
        self.register_buffer('norm1_gamma', state[prefix + 'norm1.weight'])
        self.register_buffer('norm1_beta', state[prefix + 'norm1.bias'])
        
        self.register_buffer('qkv_weight', state[prefix + 'attn.qkv.weight'])
        self.register_buffer('qkv_bias', state[prefix + 'attn.qkv.bias'])
        self.register_buffer('proj_weight', state[prefix + 'attn.proj.weight'])
        self.register_buffer('proj_bias', state[prefix + 'attn.proj.bias'])
        
        self.register_buffer('norm2_gamma', state[prefix + 'norm2.weight'])
        self.register_buffer('norm2_beta', state[prefix + 'norm2.bias'])
        
        self.register_buffer('fc1_weight', state[prefix + 'mlp.fc1.weight'])
        self.register_buffer('fc1_bias', state[prefix + 'mlp.fc1.bias'])
        self.register_buffer('fc2_weight', state[prefix + 'mlp.fc2.weight'])
        self.register_buffer('fc2_bias', state[prefix + 'mlp.fc2.bias'])

    def forward(self, x, scale, eps):
        residual = x
        
        x = vit_cuda.layernorm(x, self.norm1_gamma, self.norm1_beta, eps)
        
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, 768).permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_out = vit_cuda.flash_attn_2(q, k, v, scale)
        attn_out = F.linear(attn_out, self.proj_weight, self.proj_bias)
        
        x = residual + attn_out
        residual = x
        
        x = vit_cuda.layernorm(x, self.norm2_gamma, self.norm2_beta, eps)
        
        mlp_out, _ = vit_cuda.mlp_forward(
            x, 
            self.fc1_weight, self.fc1_bias, 
            self.fc2_weight, self.fc2_bias
        )
        
        return residual + mlp_out

class ViTCUDA(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Load pre-trained weights from standard timm model
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        state = model.state_dict()
        
        self.num_classes = num_classes
        self.scale = 1.0 / math.sqrt(64)
        self.eps = 1e-6
        
        self.register_buffer('patch_weight', state['patch_embed.proj.weight'])
        self.register_buffer('cls_token', state['cls_token'].squeeze(0))
        self.register_buffer('pos_embed', state['pos_embed'].squeeze(0))
        
        self.blocks = nn.ModuleList([ViTBlockCUDA(state, i) for i in range(12)])
        
        self.register_buffer('norm_gamma', state['norm.weight'])
        self.register_buffer('norm_beta', state['norm.bias'])
        
        self.register_buffer('head_weight', state['head.weight'])
        self.register_buffer('head_bias', state['head.bias'])

    def forward(self, x):
        x = vit_cuda.patch_embed(x, self.patch_weight)
        x = vit_cuda.pos_encoding(x, self.cls_token, self.pos_embed)
        
        for block in self.blocks:
            x = block(x, self.scale, self.eps)
            
        x = vit_cuda.layernorm(x, self.norm_gamma, self.norm_beta, self.eps)
        out = vit_cuda.classifier(x, self.head_weight, self.head_bias, self.num_classes)
        
        return out

def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def main():
    parser = argparse.ArgumentParser(description="Run Custom CUDA ViT Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()

    model = ViTCUDA().cuda()
    model.eval()

    labels = get_imagenet_labels()

    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Actual inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print(f"\nPredictions for {args.image}:")
    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        score = top5_prob[i].item()
        print(f"{i+1}: {labels[class_id]} ({score * 100:.2f}%)")

if __name__ == "__main__":
    main()