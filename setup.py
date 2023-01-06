"""ToMCubes"""
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="tomcubes",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "tomcubes_cuda",
            ["src/tomcubes_cuda.cpp", "src/tomcubes_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    install_requires=[
        "torch",
    ],
    packages=["tomcubes"],
)
