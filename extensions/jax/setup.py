from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pybind11

root = os.path.dirname(__file__)

sources = [
    os.path.join(root, 'jax_ext.cpp'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_warps.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_tc.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'fa_int8.cu'),
    os.path.join(root, '..', '..', 'mha_kernels', 'unfused.cu'),
    os.path.join(root, '..', '..', 'utils.cu'),
]

setup(
    name='jax_ext',
    ext_modules=[
        CUDAExtension(
            name='jax_ext',
            sources=sources,
            include_dirs=[os.path.join(root, '..', '..', 'include'), pybind11.get_include()],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
