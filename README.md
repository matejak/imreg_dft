imreg_dft
=========

[![Build Status](https://travis-ci.org/matejak/imreg_dft.svg?branch=master)](https://travis-ci.org/matejak/imreg_dft) [![Documentation Status](https://readthedocs.org/projects/imreg-dft/badge/?version=latest)](https://readthedocs.org/projects/imreg-dft/?badge=latest)

Overview
--------
Image registration using discrete Fourier transform.

Given two images, `imreg_dft` can calculate difference between scale, rotation and position of imaged features.

Features
--------
* Image preprocessing options (frequency filtration, image extension).
* Command-line interface (text output and/or image output).
* Python API (WIP).
* GUI (WIP).

Project facts
-------------
* The project is pure Python.
* Essentially requires `numpy` and `scipy` (`RHEL7`-safe).
* Per-commit tests and documentation (see badges under the heading).
* Originally developed by Cristoph Gohlke (University of California, USA)
* Currently developed by Matěj Týč (Brno University of Technology, CZ]
