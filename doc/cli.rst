.. _cli:

Command-line tool overview
==========================

The package contains one Python :abbr:`CLI (command-line interface)` script.
Although you are more likely to use the ``imreg_dft`` functionality from your own Python programs, you can still have some use to the ``ird`` front-end.
There are three main reasons why you would want to use it:

Quick reference
+++++++++++++++

#. Quickly find out whether it works for you, having the results (optionally) shown in a pop-up window and printed out.
   We assume you stand in the root of ``imreg_dft`` cloned from the git repo.

   .. literalinclude:: _static/examples/01-intro.txt
     :language: shell-session

   .. warning::

     Keep in mind that images have the zero coordinate (i.e. the origin) in their upper left corner!

#. You can have the results print in a defined way.
   First of all though, let's move to the examples directory:

   .. literalinclude:: _static/examples/02-print.txt
     :language: shell-session

#. Let's try something more tricky!
   The first and third examples are rotated against each other and also have a different scale.

   .. literalinclude:: _static/examples/03-bad.txt
     :language: shell-session

   Uh-oh, that didn't turn out very well, did it?
   The result is somehow better than a no-op, but highly unsatisfactory nevertheless.

   However, we have a triumph in our sleeve.
   We can force ``ir_dft`` to try to guess scale and rotation multiple times in a row.
   The correct values are -30° for the rotation and 1 / 80% = 1.25 for the scale:

   .. literalinclude:: _static/examples/04-iter.txt
     :language: shell-session

   So, four iterations are enough for a precise result!

#. And now something even more tricky - when a part of the image is cut out.
   The difference between the fourth and third sample is their mutual translation which also causes that the feature we are matching against is cut out from the fourth image.

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
   The scale and angle information is quite clear, but the translation depends on the center of scaling and the center of rotation...
   So the idea is as follows.
   Let's assume you have an image, an ``imreg_dft`` output and all you want is to perform the image transformation yourself.
   The output describes what operations to perform on the image so it is close to the template.
   All transformations are performed using ``scipy.ndimage.interpolate`` package and you need to do the following:

   i. Call the ``zoom`` function with the provided scale.
      The center of the zoom is the center of the image.

   #. Then, rotate the image using the ``rotate`` function, specifying the given angle.
      The center of the rotation is again the center of the image.

   #. Finally, translate the image using the ``shift`` function.
      Remember that the ``y`` axis is the first one and ``x`` the second one.

   #. That's it, the image should now look like the template.

#. Speaking of which, you can have the output saved to a file.
   This is handy for example if you record the same thing with different means (e.g. a cell recorded with multiple microscopes) and you want to examine the difference between them on a pixel-by-pixel basis.
   In order to be able to exploit this feature to its limits, read about ``loaders``, but you can simply try this example:

   .. literalinclude:: _static/examples/09-output.txt
     :language: shell-session

   To sum it up, the registration is a process performed with images somehow converted to grayscale (for example as the average across all color chanels).
   However, as soon as the transformation is known, an RGB image can be transformed to match the template and saved in full color.

Loaders
+++++++

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

Advanced tweaking
+++++++++++++++++

There are some extended options you can use, we will explain their meaning now:

``--lowpass``, ``--highpass``
    These two concern filtration of the image prior to the registration.
    There can be multiple reasons why to filter images:

    * One of them is filtered already due to conditions beyond your control, so by filtering them again just brings the other one on the par with the first one.
      As a side note, filtering in this case should make little to no difference.

    * A part of spectrum contains noise which you want to remove.

    * You want to filter out low frequencies since they are of no good when registering images anyway.

    The filtering works like this:

    The domain of the spectrum is a set of spatial frequencies.
    Each spatial frequency in an image is a vector with a :math:`x` and :math:`y` components.
    We norm the frequencies by stating that the highest value of a compnent is 1. 
    Next, define the *value* of spatial frequency as the (euclidean) length of the normed vector.
    Therefore the spatial frequencies of greatest values (:math:`\sqrt 2`) are (1, 1), (1, -1) etc.

    An argument to a ``--lowpass`` or ``--highpass`` option is a tuple composed of numbers between 0 and 1.
    Those relate to the value of spatial frequencies it affects.
    For example, passing ``--lowpass 0.2,0.4`` means that spatial frequencies with value ranging from 0 to 0.2 will pass and those with value higher than 0.4 won't.
    Spatial frequencies with values in-between will be progressively attenuated. 

``--filter-pcorr``
    Fitering of phase correlation applies when determining the right translation vector.
    If the image pattern is not sampled very densely (i.e. close or even below the Nyquist frequency), ripples may appear near edges in the image.
    These ripples basically interfere with the algorithm and the phase correlation filtration may overcome this problem.

``--exponent``
    When finding the right angle and scale, the highest element in an array is searched for.
    However, again due to incorrect sampling, it might not be the best guess --- for instance, this approach has the obvious flaw of being numerically unstable.
    There may be several extreme values close together and picking the center of them can be much better.
    This option plays the following role in the process:
    
    * The array is powered by the exponent.

    * The coordinates of the center of mass of the array are determined. 

    Formally: Let :math:`f(x)` be a discrete non-negative function, for instance :math:`f(0) = 3,\ f(1) = 0, f(2) = 2.99, f(3) = 1`.
    Then, the index of the greatest value is denoted by :math:`\mathrm{argmax}\, f(x) = 0`, because :math:`f(0)` is the greatest of :math:`f(x)` for all :math:`x` whete :math:`f(x)` is defined.
    The coordinate of the center of mass of :math:`f(x)` is :math:`t_f = \sum f(x_i)^c x_i / \sum f(x_i)^c`, where :math:`c` is our exponent, in case of real center of mass, :math:`c = 1`.
    The problem is that in this case, the value of :math:`\mathrm{argmax}\, f(x)` is unstable, since the difference between :math:`f(0)` and :math:`f(2)` is relatively low.
    If we consider real-world conditions, the difference could be below a fraction of the noise standard deviation.
    However, if we select a value of :math:`c = 5`, the value of corresponding :math:`t_f = 0.996`, which is just between the two highest values and not affected by :math:`f(3)`.
    And this is actually exactly what we want --- the interpolation during image transformations is not perfect and an analogous situation can occur in the spectrum.
    The center of few extreme values close together is more representative than the location of just one extreme value.

    One can generalize this to the case of 2D discrete functions and that's our case.
    Obviously, the higher the exponent is, the closer are we to picking the coordinate of the greatest array element.
    To neutralize the influence of points with low value, set the value of the exponent to greater or equal to 5.

    .. literalinclude:: _static/examples/06-exponent.txt
      :language: shell-session

    We can see that with only one iteration, setting the ``--exponent`` to ``5`` brings a more accurate result than the default value of ``'inf'`` --- the correct value is 1.25 for the scale and -30 for the angle.
    However, if we increase the number of iterations, the exponent won't make a difference any more.

``--resample``
    You can try to go for sub-pixel precision if you request resampling of the input prior to the registration.
    Resampling can be regarded as an interpolation method that is the only correct one in the case when the data are sampled correctly.
    As opposed to well-known 2D interpolation methods such as bilinear or bicubic, resampling uses the :math:`sinc(x) = sin(x) / x` function, but it is usually implemented by taking a discrete Fourier transform of the input, padding the spectrum with zeros and then performing an inverse transform.
    If you try it, results are not so great:

    .. literalinclude:: _static/examples/07-resample.txt
      :language: shell-session

    However, resampling can result in artifacts near the image edges.
    This is a known phenomenon that occurs when you have an unbounded signal (i.e. signal that goes beyond the field of view) and you manipulate its spectrum.
    Extending the image and applying a mild low-pass filter can improve things considerably.

    The first operation removes the edge artifact problem by making the opposing edges the same and making the image seamless.
    This removes spurious spatial frequencies that appear as a ``+`` pattern in the image's power spectrum.
    The second one then ensures that the power spectrum is mostly smooth after the zero-pading, which is also good.

    .. literalinclude:: _static/examples/08-resample2.txt
      :language: shell-session

    As we can see, both the scale and angle were determined extremely precisely.
    So, a warning for those who skip the ordinary text:

    .. warning::

      The ``--resample`` option offers the potential of sub-pixel resolution.
      However, when using it, be sure to start off with (let's say) ``--extend 10`` and ``--lowpass 0.9,1.1`` to exploit it.
      Then, experiment with the settings until the results look best.
    
