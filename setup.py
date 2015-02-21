# -*- coding: utf-8 -*-

import os
import sys
import setuptools as st

sys.path.append(os.path.join('src'))
import imreg_dft

reqs = open('requirements.txt', 'r').read().strip().splitlines()

longdesc = open(os.path.join('doc', 'description.rst'), 'r').read()

st.setup(
    name="imreg_dft",
    version=imreg_dft.__version__,
    author=u"Matěj Týč",
    author_email="matej.tyc@gmail.com",
    description=("Image registration utility using algorithms based on "
                 "discrete Fourier transform (DFT, FFT)"),
    license="BSD",
    url="https://github.com/matejak/imreg_dft",
    package_dir = {'': 'src'},
    packages = st.find_packages('src'),
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
