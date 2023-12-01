from setuptools import setup, Extension
from torch.utils import cpp_extension

#from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='tet_utils',
    ext_modules=[cpp_extension.CppExtension('tet_utils', ['tet_utils.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension})