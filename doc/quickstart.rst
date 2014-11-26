Quickstart guide
================

Installation
------------

Before installing ``imreg_dft``, it is good to have :ref:`dependencies <requirements>` sorted out.
``numpy`` and ``scipy`` should be installed using package managers on Linux, or `downloaded from the web <http://www.scipy.org/scipylib/download.html>`_ for OSX and Windows.

The easy(_install) way
++++++++++++++++++++++

.. warning::
  We are not on PyPi for the time being, so you can skip the entire paragraph.

You can get the package from PyPi, which means that if you have Python ``setuptools`` installed, you can install ``imreg_dft`` using ``easy_install``.
For a user (i.e. not system-wide) install, insert ``--user`` between ``easy_install`` and ``imreg_dft``.
`User install <https://pythonhosted.org/setuptools/easy_install.html#custom-installation-locations>`_ does not require administrator priviledges, but you may need to add the installation directories to your system path, otherwise the `ird` script won't be visible.

.. code-block:: shell-session

  [user@linuxbox ~]$ easy_install imreg_dft

If you have ``pip`` installed, you can `use it <https://pip.pypa.io/en/latest/user_guide.html#installing-packages>`_ instead of ``easy_install``.

The source way (also easy)
++++++++++++++++++++++++++

The other means is to check out the repository and install it locally (or even run ``imreg_dft`` without installation).
You will need the ``git`` version control system to obtain the source code:

.. code-block:: shell-session

  [user@linuxbox ~]$ git clone https://github.com/matejak/imreg_dft.git
  [user@linuxbox ~]$ cd imreg_dft
  [user@linuxbox imreg_dft]$ python setup.py install

As with other Python packages, there is a ``setup.py``.
To install ``imreg_dft``, run ``python setup.py install`` (add ``--user`` argument after ``install`` to perform user install).

If you want to try ``imreg_dft`` without installing it, feel free to do so.
The package is in the ``src`` directory.

Quickstart
----------

A succesful installation means that:

* The Python interpreter can import ``imreg_dft``.
* There is the ``ird`` script available to you, e.g. running ``ird --version`` should not end by errors of any kind.

Python examples
+++++++++++++++

The following examples are located in the ``resources/code`` directory of the project repository as ``similarity.py`` and ``translation.py``.
You can launch them from their location once you have installed ``imreg_dft`` to observe the output.

The full-blown similarity function that returns parameters (and the transormed image):

.. literalinclude:: ../resources/code/similarity.py
    :language: python

Or just the translation:

.. literalinclude:: ../resources/code/translation.py
    :language: python

Command-line script examples
++++++++++++++++++++++++++++

Please see the :ref:`corresponding section <cli>` that is full of examples.
