.. _utils:

Utility scripts
===============

There are two scripts that complement the main :ref:`ird script <cli>`.
Those are:

* Transfomation tool --- good if you know relation between the template and subject and you just want to transform the subject in the same way as the ``ird`` tool.
* Incpection tool --- intended for gaining insight into the phase correlation as such.
  Especially handy in cases when something goes wrong or when you want to gain insight into the phase correlation process.

Transormation tool
------------------

The classical use case of phase correlation is a situation when you have the subject, the template, and your goal is to transform the subject in a way that it matches the template.
The transformation parameters are unknown, and the purpose of phase correlation is to compute them.

So the use case consists of two sub-cases:

* Compute relation of the two images, and
* transform the subject according to the result of the former.

The ``imreg_dft`` project enables you to do all using the ``ird`` script, but those two steps can be split --- the first can be done by ``ird``, whereas the second by ``ird-tform``.
The transform parameters can be specified as an argument, or they can be read from stdin.

Therefore, those two one-liners are equivalent --- the file ``subject-tformed.png`` is the rotated subject ``sample3.png``, so it matches the template ``sample1.png``.
Also note that the ``ird`` script alone will do the job faster.

.. code-block:: shell-session

   [user@linuxbox examples]$ ird sample1.png sample3.png -o subject-tformed.png
   [user@linuxbox examples]$ ird sample1.png sample3.png --print-result | ird-tform sample3.png --template sample1.png subject-tformed.png

Technically, the output of ``ird`` alone should be identical to ``ird-tform``.

Inspection tool
---------------

The phase correlation method is built around the Fourier transform and some relatively simple concepts around it.

Although the phase correlation is an image registration technique that is highly regarded by field experts, it may produce unwanted results.
In order to find out what is going on, you can request visualizations of various phases of the registration process.

