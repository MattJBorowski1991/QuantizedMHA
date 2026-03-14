from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
import glob

root = os.path.dirname(os.path.abspath(__file__))

# Get kernel selection from KERNEL environment variable (default: all)
kernel_choice = os.environ.get('KERNEL', 'all').strip()

sources = [
    os.path.join(root, 'torch_ext.cpp'),
    os.path.join(root, '..', '..', 'drivers', 'main.cu'),
    os.path.join(root, '..', '..', 'inputs', 'data.cu'),
    os.path.join(root, '..', '..', 'utils', 'utils.cu'),
    os.path.join(root, '..', '..', 'utils', 'verify.cu'),
]

# Define available kernels
kernels = {
    'fa': 'mha_kernels/fa.cu',
    'fa_tc_int8_a': 'mha_kernels/fa_tc_int8_a.cu',
    'fa_tc_int8_b': 'mha_kernels/fa_tc_int8_b.cu',
    'fa_tc_v1a': 'mha_kernels/fa_tc_v1a.cu',
    'fa_tc_v1b': 'mha_kernels/fa_tc_v1b.cu',
    'fa_tc_v2': 'mha_kernels/fa_tc_v2.cu',
    'fa_tc_v2a': 'mha_kernels/fa_tc_v2a.cu',
    'fa_tc_v2b': 'mha_kernels/fa_tc_v2b.cu',
    'unfused': 'mha_kernels/unfused.cu',
}

# Add kernels based on selection
if kernel_choice == 'all':
    print(f"[setup.py] Compiling all {len(kernels)} kernels")
    for k, path in kernels.items():
        sources.append(os.path.join(root, '..', '..', path))
elif kernel_choice in kernels:
    print(f"[setup.py] Compiling only kernel: {kernel_choice}")
    sources.append(os.path.join(root, '..', '..', kernels[kernel_choice]))
else:
    raise ValueError(f"[setup.py] Unknown kernel '{kernel_choice}'\n"
                     f"Options: all, {', '.join(sorted(kernels.keys()))}\n"
                     f"Usage: KERNEL=<name> python setup.py build_ext --inplace")


class CopyBuildExt(BuildExtension):
    """Custom build extension that copies .so to current directory"""
    def build_extension(self, ext):
        super().build_extension(ext)
        # Copy .so file to current directory
        if self.inplace:
            build_lib = self.get_lib_include_dirs()[0]
            # Find all .so files in build directory
            for so_file in glob.glob(os.path.join(self.build_lib, '*.so')):
                dest = os.path.join(root, os.path.basename(so_file))
                print(f"[setup.py] Copying {so_file} -> {dest}")
                shutil.copy2(so_file, dest)


setup(
    name='torch_ext',
    ext_modules=[
        CUDAExtension(
            name='torch_ext',
            sources=sources,
            include_dirs=[os.path.join(root, '..', '..', 'include')],
        )
    ],
    cmdclass={'build_ext': CopyBuildExt},
)
