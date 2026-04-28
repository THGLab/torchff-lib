import os, glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if 'CXX' not in os.environ:
    os.environ['CXX'] = 'g++'
    print("Set C++ compiler: g++")


def build_cuda_extension(name, exclude_files=list()):
    sources = []
    for file in glob.glob(os.path.join(os.path.dirname(__file__), f"csrc/{name}/*")):
        if (file.endswith('.cpp') or file.endswith('.cu') or file.endswith('.c') or file.endswith('.C')) and (os.path.basename(file) not in exclude_files):
            sources.append(file)
    
    return CUDAExtension(
        name=f'torchff_{name}',
        sources=sources,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        },
        include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
    )


setup(
    name='torchff',
    classifiers=[
        'Development Status :: Beta',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.12',
    ],
    packages=find_packages(exclude=['csrc', 'tests', 'docs'], include=['torchff*']),
    ext_modules=[
        build_cuda_extension('bond'),
        build_cuda_extension('angle'),
        build_cuda_extension('torsion'),
        build_cuda_extension('vdw'),
        build_cuda_extension('dispersion'),
        build_cuda_extension('slater'),
        build_cuda_extension('coulomb'),
        build_cuda_extension('multipoles'),
        build_cuda_extension('amoeba'),
        build_cuda_extension('ewald', ['ewald_optimized.cu']),
        build_cuda_extension('pme'),
        build_cuda_extension('cmm'),
        # only compile the fused nonbonded atom-pair kernel in csrc/nonbonded
        CUDAExtension(
            name='torchff_nb',
            sources=['csrc/nonbonded/nonbonded_interface.cpp',
                     'csrc/nonbonded/nonbonded_atom_pairs_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
        ),
        CUDAExtension(
            name='torchff_nblist',
            sources=['csrc/nblist/nblist_interface.cpp',
                     'csrc/nblist/nblist_nsquared_cuda.cu',
                     'csrc/nblist/nblist_clist_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
