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
import imreg_dft

reqsfname = os.path.join(SETUPDIR, 'requirements.txt')
reqs = open(reqsfname, 'r', encoding='utf-8').read().strip().splitlines()

descfname = os.path.join(SETUPDIR, 'doc', 'description.rst')
longdesc = open(descfname, 'r', encoding='utf-8').read()

st.setup(
    name="imreg_dft",
    version=imreg_dft.__version__,
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
        ],
    },
    install_requires=reqs,
    extras_require={
        'plotting':  ["matplotlib>=1.2"],
        'loading images': ["pillow>=2.2"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
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
