import timm
m = timm.create_model('vit_base_patch16_224', pretrained=True)
print('has_bias', m.patch_embed.proj.bias is not None)
print('bias_shape', None if m.patch_embed.proj.bias is None else tuple(m.patch_embed.proj.bias.shape))
