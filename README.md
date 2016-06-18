imreg_dft
=========

[![Latest Version on PyPi](http://img.shields.io/pypi/v/imreg_dft.svg)](https://pypi.python.org/pypi/imreg_dft)
[![Build Status](https://travis-ci.org/matejak/imreg_dft.svg?branch=master)](https://travis-ci.org/matejak/imreg_dft)
[![Documentation Status](https://readthedocs.org/projects/imreg-dft/badge/?version=latest)](https://readthedocs.org/projects/imreg-dft/?badge=latest)

Overview
--------
Image registration using discrete Fourier transform.

Given two images, `imreg_dft` can calculate difference between scale, rotation and position of imaged features.
Given you have the requirements, you can start aligning images in about five minutes!
Check the documentation on [readthedocs.org](http://imreg-dft.readthedocs.org/en/latest/quickstart.html) (bleeding-edge) or [pythonhosted.org](http://pythonhosted.org//imreg_dft/) (with images).

If you are a part of the education process that has something to do with phase correlation (doesn't matter whether you are a student or a teacher), the `ird-show` utility that is part of the `imreg_dft` project can help you understand (or explain) how the phase correlation process works.
If you are a researcher and you are having problems with the method on your data, you can use `ird-show` to find out what causes your problems quite easily.

Features
--------
* Image pre-processing options (frequency filtration, image extension).
* Under-the-hood options exposed (iterations, phase correlation filtration).
* Visualization of various stages of the image registration (so you can more easily find out how it works or what went wrong).
* Command-line interface for image registration (`ird` - text output and/or image output), for image transformation (`ird-tform`, cooperates with `ird`) and inspection (`ird-show`).
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
