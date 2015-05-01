.. imreg_dft documentation master file, created by
   sphinx-quickstart2 on Sun Oct 12 15:24:34 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imreg_dft's documentation!
=====================================

General overview
----------------

``imreg_dft`` implements DFT\ [*]_ -based technique for translation, rotation and scale-invariant image registration.

In plain language, ``imreg_dft`` implements means of calculating translation, rotation and scale variation between two images.
It doesn't work with those images directly, but it works with their spectrum, using the log-polar transformation.
The algorithm is described in [1]_ and possibly also in [2]_ and [3]_.

.. [*] DFT stands for Discrete Fourier Transform.
   Usually the acronym FFT (Fast Fourier Transform) is used in this context, but this is incorrect.
   DFT is the name of the operation, whereas FFT is just one of possible algorithms that can be used to calculate it.

.. figure:: _build/images/big.*

   The template (a), sample (b) and registered sample (c).
   This is the actual output of :ref:`sample in the cli section <sample-intro>`

:Authors:
  - `Matěj Týč <https://github.com/matejak>`_
  - `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  - Brno University of Technology, Brno, Czech Republic
  - Laboratory for Fluorescence Dynamics, University of California, Irvine

:Copyright:
  - 2014-2015, Matěj Týč
  - 2011-2014, Christoph Gohlke

.. _requirements:

Requirements
++++++++++++

See the ``requirements.txt`` file in the project's root for the exact specification.
Generally, you will need ``numpy`` and ``scipy`` for the core algorithm functionality.

Optionally, you may need:

 - ``pillow`` for loading data from image files,
 - ``matplotlib`` for displaying image output,
 - ``pyfftw`` for better performance.

Quickstart
++++++++++

Head for the :ref:`corresponding section of the documentation <quickstart>`.
Note that you can generate the documentation yourself!

1. Install the package by running ``python setup.py install`` in the project root.
#. Install packages that are required for the documentation to compile (see the ``requirements_docs.txt`` file.
#. Go to the ``doc`` directory and run ``make html`` there.
   The documentation should appear in the ``_build`` subfolder, so you may open ``_build/html/index.html`` with your web browser to see it.

Notes
+++++

The API and algorithms are quite good, but help is appreciated.
``imreg_dft`` uses `semantic versioning <http://semver.org/>`_, so backward compatibility of any kind will not break across versions with the same major version number.

``imreg_dft`` is based on the `code <http://www.lfd.uci.edu/~gohlke/code/imreg.py.html>`_ by Christoph Gohlke.

References
++++++++++

.. [1] An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
.. [2] An IDL/ENVI implementation of the FFT-based algorithm for automatic
    image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
    Computers & Geosciences, 29, 1045-1055, 2003.
.. [3] Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
    RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart.rst
   cli.rst
   cli-advanced.rst
   api.rst
   devel.rst
   changelog.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

