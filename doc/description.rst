General overview
================

``imreg_dft`` implements means of guessing translation, rotation and scale variation between two images.
It doesn't work with those images directly, but it works with their spectrum (DFT using FFT), and its log-polar transformation [1]_.

Basically, if you want to align two images that have different scale and are rotated and shifted against each other, ``imreg_dft`` is the tool you want to check out.
`Get started <Quickstart>`_ in five minutes and see how it works for you!

:Authors:
  - `Matěj Týč <https://github.com/matejak>`_
  - `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  - Laboratory for Fluorescence Dynamics, University of California, Irvine
  - Brno University of Technology, Brno, Czech Republic

:Copyright:
  - 2011-2014, Christoph Gohlke
  - 2014-2015, Matěj Týč

.. _requirements:
 
Requirements
------------
Generally, you will need ``numpy`` and ``scipy`` for the algorithm functionality and ``matplotlib`` for plotting.
For the command-line tool, reading images is useful, so make sure you have ``pillow`` (or ``PIL``, which is deprecated).

Quickstart
----------

Check the documentation on `readthedocs.ort <http://imreg-dft.readthedocs.org/en/latest/quickstart.html>`_ (bleeding-edge) or `pythonhosted.org <http://pythonhosted.org/imreg_dft/quickstart.html>`_ (with images).
Or even better, generate the documentation yourself! 

1. Install the package by running ``python setup.py install`` in the project root.
#. Install packages that are required for the documentation to compile (see the ``requirements_docs.txt`` file.
#. Go to the ``doc`` directory and run ``make html`` there.
   The documentation should appear in the ``_build`` subfolder, so you may open ``_build/html/index.html`` with your web browser to see it.

Notes
-----

``imreg_dft`` is based on the `code <http://www.lfd.uci.edu/~gohlke/code/imreg.py.html>`_ by Christoph Gohlke.

References
----------
.. [1] An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
