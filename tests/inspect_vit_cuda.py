import torch
import vit_cuda

names = sorted([n for n in dir(vit_cuda) if 'flash' in n or 'attn' in n])
print('matching names:', names)
print('module file:', getattr(vit_cuda, '__file__', None))
print('has flash_attn_2:', hasattr(vit_cuda, 'flash_attn_2'))
