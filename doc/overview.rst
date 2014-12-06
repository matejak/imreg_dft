General overview
================

FFT based image registration.

Implements an FFT-based technique for translation, rotation and scale-invariant
image registration [1].

:Authors:
  - `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_
  - `Matěj Týč <https://github.com/matejak>`_

:Organization:
  - Laboratory for Fluorescence Dynamics, University of California, Irvine
  - Brno University of Technology, Brno, Czech Republic

:Copyright:
  - 2011-2014, Christoph Gohlke
  - 2014, Matěj Týč


.. _requirements:
 
Requirements
------------
See the ``requirements.txt`` file in the project's root for the exact specification.
Generally, you will need ``numpy`` and ``scipy`` for the algorithm functionality and ``matplotlib`` for plotting.

Quickstart
----------

`Read the docs <http://imreg-dft.readthedocs.org>`_, the full-blown quickstart is there.
Disregard this section if you actually are reading the docs (and not the project's webpage, where this text also appears).

Or even better, generate the documentation yourself! 

1. Install the package by running ``python setup.py install`` in the project root.
#. Install packages that are required for the documentation to compile (see the ``requirements_docs.txt`` file.
#. Go to the ``doc`` directory and run ``make html`` there.
   The documentation should appear in the ``_build`` subfolder, so you may open ``_build/html/index.html`` with your web browser to see it.

Notes
-----
The API and algorithms are quite good, but help is appreciated.
``imreg_dft`` uses `semantic versioning <http://semver.org/>`_, so backward compatibility of any kind will not break across versions with the same major version number.

``imreg_dft`` is based on the `code <http://www.lfd.uci.edu/~gohlke/code/imreg.py.html>`_ by Christoph Gohlke.

References
----------
(1) An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
(2) An IDL/ENVI implementation of the FFT-based algorithm for automatic
    image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
    Computers & Geosciences, 29, 1045-1055, 2003.
(3) Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
    RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.
