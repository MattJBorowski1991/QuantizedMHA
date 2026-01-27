from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

root = os.path.dirname(__file__)

sources = [
    os.path.join(root, 'torch_ext.cpp'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_warps.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_tc.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_int8.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'unfused.cu'),
    os.path.join(root, '..', '..', 'utils.cu'),
]

setup(
    name='torch_ext',
    ext_modules=[
        CUDAExtension(
            name='torch_ext',
            sources=sources,
            include_dirs=[os.path.join(root, '..', '..', 'include')],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
