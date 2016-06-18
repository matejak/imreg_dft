User-centric changelog
======================

**2.0.0 --- 2016-06-19**

 * Preliminary support for sub-pixel transformation detection (watch `#10 <https://github.com/matejak/imreg_dft/issues/10>`_).
 * Changed interface for the ``imreg_dft.imreg.translation`` function (it returns a dict).
 * New script: ``ird-tform`` transforms an image if given a transformation (closes `#19 <https://github.com/matejak/imreg_dft/issues/19>`_).
 * New script: ``ird-show`` allows in-depth visual insight into a phase correlation operation (closes `#20 <https://github.com/matejak/imreg_dft/issues/20>`_).

**1.0.5 --- 2015-05-02**

 * Fixed project documentation typos, added the ``AUTHORS`` file.
 * Added support for ``pyfftw`` for increased performance.
 * Improved integration with MS Windows.
 * Fixed an install bug (closes `#18 <https://github.com/matejak/imreg_dft/issues/18>`_) that occured when dependencies were not met at install-time.
 * Added documentation for Python constraint interface (closes `#15 <https://github.com/matejak/imreg_dft/issues/15>`_).

**1.0.4 --- 2015-03-03**

 * Increased robustness of the tiling algorithm (i.e. when matching small subjects against large templates).
 * Improved regression tests.
 * Fixed project description typo.

**1.0.3 --- 2015-02-22**

  * Fixed the ``MANIFEST.in`` so the package is finally ``easy_install``-able.
  * Added the release check script stub.
  * Updated install docs.

**1.0.2 --- 2015-02-21**

  * Documentation and setup.py fixes.

**1.0.1 --- 2015-02-19**
  
  * Real debut on PyPi.
  * Fixed some minor issues with setup.py and docs.

**1.0.0 --- 2015-02-19**
  
  Beginning of the changelog.

  * Debut on PyPi
