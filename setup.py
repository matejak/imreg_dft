# -*- coding: utf-8 -*-

import os
import sys
import setuptools as st
from io import open

# Fix so that the setup.py usage is CWD-independent
SETUPDIR = os.path.abspath(os.path.dirname(__file__))
SETUPDIR = os.path.dirname(__file__)
PKGDIR = os.path.join(SETUPDIR, 'src')

sys.path.append(PKGDIR)

reqsfname = os.path.join(SETUPDIR, 'requirements.txt')
reqs = open(reqsfname, 'r', encoding='utf-8').read().strip().splitlines()

# get version from __init__.py without importing
versfname = os.path.join(PKGDIR, 'imreg_dft', '__init__.py')
init_lines = open(versfname).readlines()
version_line = list(filter(lambda x: '__version__' in x, init_lines))[0]
version = version_line.split('"')[1]

descfname = os.path.join(SETUPDIR, 'doc', 'description.rst')
longdesc = open(descfname, 'r', encoding='utf-8').read()

st.setup(
    name="imreg_dft",
    version=version,
    author=u"Matěj Týč",
    author_email="matej.tyc@gmail.com",
    description=("Image registration utility using algorithms based on "
                 "discrete Fourier transform (DFT, FFT)"),
    license="BSD",
    url="https://github.com/matejak/imreg_dft",
    package_dir = {'': PKGDIR},
    packages = st.find_packages(PKGDIR),
    entry_points = {
        'console_scripts': [
           'ird = imreg_dft.cli:main',
           'ird-tform = imreg_dft.tform:main',
           'ird-show = imreg_dft.show:main',
        ],
    },
    install_requires=reqs,
    extras_require={
        'plotting':  ["matplotlib>=1.2"],
        'loading images': ["pillow>=2.2"],
        'better performance': ["pyfftw>=0.9"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    long_description=longdesc,
    zip_safe=True,
)
