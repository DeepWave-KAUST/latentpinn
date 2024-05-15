import os
from glob import glob
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Repository for DW0037: Wavefield solutions using a physics-informed neural network as a function of velocity.'

from setuptools import setup

setup(
    name="latentpinn",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'generative models',
              'deep learning',
              'tomography',
              'efwi',
              'seismic'],
    author='Mohammad H. Taufik, Xinquan Huang, Tariq Alkhalifah',
    author_email='mohammad.taufik@kaust.edu.sa, xinquan.huang@kaust.edu.sa, tariq.alkhalifah@kaust.edu.sa',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'setuptools_scm',
    ],
)
