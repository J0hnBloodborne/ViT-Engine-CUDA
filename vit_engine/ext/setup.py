from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vit_cuda',
    ext_modules=[
        CUDAExtension(
            name='vit_cuda',
            sources=[
                'binding.cpp',
                'patch_embed.cu',
                'pos_embed.cu',
                'attention.cu',
                'mlp.cu',
                'layernorm.cu',
                'classifier.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
