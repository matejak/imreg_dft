imreg_dft
=========

[![Latest Version on PyPi](https://pypip.in/version/imreg_dft/badge.svg)](https://pypi.python.org/pypi/imreg_dft)
[![Build Status](https://travis-ci.org/matejak/imreg_dft.svg?branch=master)](https://travis-ci.org/matejak/imreg_dft)
[![Documentation Status](https://readthedocs.org/projects/imreg-dft/badge/?version=latest)](https://readthedocs.org/projects/imreg-dft/?badge=latest)

Overview
--------
Image registration using discrete Fourier transform.

Given two images, `imreg_dft` can calculate difference between scale, rotation and position of imaged features.
Given you have the requirements, you can start aligning images in about five minutes!
Check the documentation on [readthedocs.ort](http://imreg-dft.readthedocs.org/en/latest/quickstart.html) (bleeding-edge) or [pythonhosted.org](http://pythonhosted.org//imreg_dft/) (with images).

Features
--------
* Image pre-processing options (frequency filtration, image extension).
* Under-the-hood options exposed (iterations, phase correlation filtration).
* Command-line interface (text output and/or image output).
* Documented Python API with examples.
* Permissive open-source license (3-clause BSD).

Project facts
-------------
* The project is written in pure Python.
* Essentially requires only `numpy` and `scipy` (`RHEL7`-safe).
* Includes quickstart documentation and example data files.
* Per-commit tests and documentation (see badges under the heading).
* Originally developed by Christoph Gohlke (University of California, Irvine, USA)
* Currently developed by Matěj Týč (Brno University of Technology, CZ)
