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

Notes
-----
The API and algorithms are quite good, but help is appreciated.
``imreg_dft`` uses semantic versioning, so backward compatibility of any kind will not break across versions with the same major version number.

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
