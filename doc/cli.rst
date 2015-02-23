.. _cli:

Command-line tool overview
==========================

The package contains one Python :abbr:`CLI (command-line interface)` script.
Although you are more likely to use the ``imreg_dft`` functionality from your own Python programs, you can still have some use to the ``ird`` front-end.

There are these main reasons why you would want to use it:

* Quickly learn whether ``imreg_dft`` is relevant for your use case.
* Quickly tune the advanced image registration settings.
* Use ``ird`` in a script and process batches of images instead of one-by-one.
  (``ird`` can't do it, but you can call it in the script, possibly using `GNU Paralell <http://www.gnu.org/software/parallel>`_.)

General considerations
----------------------

Generally, you call ``ird`` with two images, the first one being the ``template`` and the second one simply the ``subject``.
If you are not sure about other program options, run it with ``--help`` or ``--usage`` arguments:

   .. literalinclude:: _static/examples/13-usage.txt
     :language: shell-session
     :end-before: positional
     :append: ...

The notation ``[foo]`` means that specifying content of brackets (in this case ``foo``) is optional.
For example, let's look at a part of help ``[--angle MEAN[,STD]]``.
The outer square bracket means that specifying ``--angle`` is optional.
However, if you specify it, you have to include the mean value as an argument, i.e. ``--angle 30``.
The inner square brackets then point out that you may also specify a standard deviation, in which case and you separate it from the mean using comma: ``--angle 30,1.5``.
There are sanity checks present, so you will be notified if you commit a mistake.

So only the input arguments are obligatory.
Typically though, you will want to add arguments to also get the result:

#. Take an instant look at the registration result --- use the ``--show`` argument.
#. Learn the registration parameters: Use the ``--print-result`` argument (explained in greater detail below).
#. Save the transformed subject: Use the ``--output`` argument.

The image registration works best if you have images that have features in the middle and their background is mostly uniform.
If you have a section of a large image and you want to use registration to identify it, :ref:`most likely, you will not succeed <weak-big>`.

For more exhaustive list of known limitation, see the section Caveats_.

Quick reference
---------------

.. _sample-intro:

#. Quickly find out whether it works for you, having the results (optionally) shown in a pop-up window and printed out.
   We assume you stand in the root of ``imreg_dft`` cloned from the git repo (or :ref:`downloaded from the web <source-files>`).

   .. literalinclude:: _static/examples/01-intro.txt
     :language: shell-session

   The output tells you what has to be done with the ``subject`` so it looks like the ``template``.

   .. warning::

     Keep in mind that images have the zero coordinate (i.e. the origin) in their upper left corner!

#. You can have the results print in a defined way.
   First of all though, let's move to the examples directory:

   .. literalinclude:: _static/examples/02-print.txt
     :language: shell-session

   You can get an up-to-date listing of possible values you can print using the help argument.
   Generally, you can get back the values as well as confidence interval half-widths that have a ``D`` prefix.
   For example there is ``angle`` and ``Dangle``; in case that the method doesn't fail misreably, the true angle will not differ from ``angle`` more than over ``Dangle``.

#. Let's try something more tricky!
   The first and third examples are rotated against each other and also have a different scale.

   .. literalinclude:: _static/examples/03-bad.txt
     :language: shell-session

#. And now something even more tricky - when a part of the subject is cut out.
   The difference between the fourth and third image is their mutual translation which also causes that the feature we are matching against is cut out from the fourth one.

   Generally, we have to address the this
   The ``--extend`` option here serves exactly this purpose.
   It extends the image by a given amount of pixels (on each side) and it tries to blur the cut-out image beyond its original border.
   Although the blurring might not look very impressive, it makes a huge difference for the image's spectrum which is used for the registration.
   So let's try:

   .. literalinclude:: _static/examples/05-extend.txt
     :language: shell-session

   As we can see, the result is correct.

   Extension can occur on-demand when the scale change or rotation operations result in image size growth.
   However, whether this will occur or not is not obvious, so it is advisable to specify the argument manually.
   In this example (and possibly in the majority of other examples) specifying the option manually is not needed.

   .. warning::

     If the image extension by blurring is very different from how the image really looks like, the image registration will fail.
     Don't use this option until you become sure that it improves the registration quality.

#. Buy what do we actually get on output?
   You may wonder what those numbers mean.
   *The output tells you what has to be done with the ``image`` so it looks like the ``template``.*
   The scale and angle information is quite clear, but the translation depends on the center of scaling and the center of rotation...

   So the idea is as follows --- let's assume you have an image, an ``imreg_dft`` print output and all you want is to perform the image transformation yourself.
   The output describes what operations to perform on the image so it is close to the template.
   All transformations are performed using `scipy.ndimage.interpolate <http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.interpolation>`_ package and you need to do the following:

   i. Call the ``zoom`` function with the provided scale.
      The center of the zoom is the center of the subject.

   #. Then, rotate the subject using the ``rotate`` function, specifying the given angle.
      The center of the rotation is again the center of the subject.

   #. Finally, translate the subject using the ``shift`` function.
      Remember that the ``y`` axis is the first one and ``x`` the second one.

   #. That's it, the subject should now look like the template.

#. Speaking of which, you can have the output saved to a file.
   This is handy for example if you record the same thing with different means (e.g. a cell recorded with multiple microscopes) and you want to examine the difference between them on a pixel-by-pixel basis.
   In order to be able to exploit this feature to its limits, read about ``loaders``, but you can simply try this example:

   .. literalinclude:: _static/examples/09-output.txt
     :language: shell-session

   To sum it up, the registration is a process performed with images somehow converted to grayscale (for example as the average across all color chanels).
   However, as soon as the transformation is known, an RGB image can be transformed to match the template and saved in full color.

Loaders
-------

``ird`` can support a wide variety of input formats.
It uses an abstract means of how to load and save an image.

To cut the long story short --- you probably want to autodetection of how to load an image based on the file extension.
The list of available loaders is obtained by passing the ``--help-loader``.
To inquire about meaning of individual options, also specify a loader on the same command-line, e.g. pass ``--loader pil``.

To pass an option to change loader properties pass a ``--loader-opts`` argument.
It accepts comma-separated ``option name=value`` pairs, so for example the ``mat`` loader understands ``--loader-opts in=imgdata,out=timgdata``.
Note that all loaders have access to passed options.

The loaders concept functionality is limited by now, but it can be extended easily by writing code.
See the :ref:`developer documentation <loaders_devel>` to learn the background.
If you miss some functionality, you are kindly invited to create a pull request!

Caveats
-------

There are known weak points of ``ird``.
Although they are explained in other places in the text, we sum them up here:

.. _weak-extend:

Extending images.
    Due to the fact that the spatial frequencies spectrum is used, the border of images are become important.
    We address it here by extending the image, but it often doesn't work well.

.. _weak-subpixel:

Sub-pixel resolution.
    This is a tricky matter.
    Since the frequency spectrum is used, neither linear or cubic interpolation will produce the right result.
    You can try resampling if you are after the sub-pixel precision, but beware --- you have to have correctly sampled (i.e. not `undersampled <http://en.wikipedia.org/wiki/Undersampling>`_) input for it to work.

.. _weak-big:

Big template.
   If the template presents a wider field of view than the image, you may or may not be successful when using the ``--tile`` option.
   The current implementation is flaky.

.. _weak-succ:

Success value.
   The ``Success`` that is reported has an unclear meaning.
   And its behavior is also quite dodgy.
