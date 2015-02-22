Quickstart guide
================

Installation
------------

Before installing ``imreg_dft``, it is good to have :ref:`dependencies <requirements>` sorted out.
``numpy`` and ``scipy`` should be installed using package managers on Linux, or `downloaded from the web <http://www.scipy.org/scipylib/download.html>`_ for OSX and Windows.

The easy(_install) way
++++++++++++++++++++++

You can get the package from PyPi, which means that if you have Python ``setuptools`` installed, you can install ``imreg_dft`` using ``easy_install``.
For a user (i.e. not system-wide) install, insert ``--user`` between ``easy_install`` and ``imreg_dft``.
`User install <https://pythonhosted.org/setuptools/easy_install.html#custom-installation-locations>`_ does not require administrator priviledges, but you may need to add the installation directories to your system path, otherwise the `ird` script won't be visible.

.. code-block:: shell-session

  [user@linuxbox ~]$ easy_install imreg_dft

If you have ``pip`` installed, you can `use it <https://pip.pypa.io/en/latest/user_guide.html#installing-packages>`_ instead of ``easy_install``.
``pip`` even allows you to install from the source code repository:

.. code-block:: shell-session

  [user@linuxbox ~]$ pip install git+https://github.com/matejak/imreg_dft.git

The source way (also easy)
++++++++++++++++++++++++++

The other means is to check out the repository and install it locally (or even run ``imreg_dft`` without installation).
You will need the ``git`` version control system to obtain the source code:

.. code-block:: shell-session

  [user@linuxbox ~]$ git clone https://github.com/matejak/imreg_dft.git
  [user@linuxbox ~]$ cd imreg_dft
  [user@linuxbox imreg_dft]$ python setup.py install
  ...

As with other Python packages, there is a ``setup.py``.
To install ``imreg_dft``, run ``python setup.py install``.
Add the ``--user`` argument after ``install`` to perform user (i.e. not system-wide) install.
As stated in the previous paragraph, the `user install <https://pythonhosted.org/setuptools/easy_install.html#custom-installation-locations>`_ does not require administrator priviledges, but you may need to add the installation directories to your system path, otherwise the `ird` script won't be visible.

If you want to try ``imreg_dft`` without installing it, feel free to do so.
The package is in the ``src`` directory.

.. _quickstart:

Quickstart
----------

A succesful installation means that:

* The Python interpreter can import ``imreg_dft``.
* There is the ``ird`` script available to you, e.g. running ``ird --version`` should not end by errors of any kind.

.. _source-files:

.. note::

   If you have installed the package using ``pip`` or ``easy_install``, you don't have the example files, images nor test files.
   To get them, download the source archive from `PyPi <https://pypi.python.org/pypi/imreg_dft/>`_ or release archive from `Github <https://github.com/matejak/imreg_dft/releases>`_ and unpack them.

.. _py_examples:

Python examples
+++++++++++++++

The following examples are located in the ``resources/code`` directory of the project repository :ref:`or its source tree <source-files>` as ``similarity.py`` and ``translation.py``.
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

Do not forget
-------------

These steps should go before the :ref:`quickstart` section, but they come now as this is a quickstart guide.

Tests
+++++

If you have :ref:`downloaded the source files <source-files>`, you can run tests after installation.
There are now unit tests and regression tests.
You can execute them by going to the ``tests`` subdirectory and running

.. literalinclude:: _static/examples/10-testing_help.txt
    :language: shell-session

If you have the ``coverage`` module installed, you also have a ``coverage`` (or perhaps ``coverage2``) scripts in your path.
You can declare that and therefore have the tests ran with coverage support:

.. code-block:: shell-session

  [user@linuxbox tests]$ make check COVERAGE=coverage2
  ...

In any way, if you see something like

.. code-block:: shell-session

  [user@linuxbox tests]$ make check
  ...
  make[1]: Leaving directory '/home/user/imreg_dft/tests'
  * * * * * * * * * * * * * * * * * * * * *
   Rejoice, tests have passed successfully!
  * * * * * * * * * * * * * * * * * * * * *

it is a clear sign that there indeed was no error encountered during the tests at all.

Documentation
+++++++++++++

Although you can read the documentation on `readthedocs.org <http://imreg-dft.readthedocs.org/en/latest/index.html>`_ (bleeding-edge) and `pythonhosted.org <http://pythonhosted.org//imreg_dft/>`_ (with images), you can generate your own easily.
You just have to check out the ``requirements_docs.txt`` file at the root of the project and make sure you have all modules that are mentioned there.
You also need to have ``imreg_dft`` installed prior documentation generation.

So, be sure to have :ref:`the source files <source-files>`.
In the source tree, go to the ``doc`` directory there and run ``make html`` or ``make latexpdf`` there.
