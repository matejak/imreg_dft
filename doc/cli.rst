Command-line tool overview
==========================

The package contains one Python :abbr:`CLI (command-line interface)` script.
Although you are more likely to use the ``imreg_dft`` functionality from your own Python programs, you can still have some use to the ``ird`` front-end.
There are three main reasons why you would want to use it:

Quick reference
+++++++++++++++

#. Quickly find out whether it works for you, having the results shown in a pop-up window and printed out.
   We assume you stand in the root of ``imreg_dft`` cloned from the git repo.

   .. code-block:: shell-session

     [user@linuxbox ir_dft]$ ird resources/examples/sample1.png resources/examples/sample2.png --show --print-result
     scale: 1.000000
     angle: 0.000000
     shift: -19, 79

   .. warning::

     Remember that images have the zero coordinate (i.e. the origin) in their upper left corner!

#. You can have the results print in a defined way.
   First of all though, let's move to the examples directory:

   .. code-block:: shell-session

     [user@linuxbox ir_dft]$ cd resources/examples
     [user@linuxbox examples]$ ird sample1.png sample2.png --print-result --print-format 'translation:%(tx)d,%(ty)d\n'
     translation:-19,79


#. Let's try something tricky - the first and third example!

   .. code-block:: shell-session

     [user@linuxbox examples]$ ird sample1.png sample3.png --show

   Uh-oh, that didn't turn out very well, did it?
   The result is somehow better than the input, but highly unsatisfactory nevertheless.

   However, we have a triumph in our sleeve.
   We can force ``ir_dft`` to try to guess scale and rotation multiple times in a row.
   The correct values are -30° for the rotation and 1 / 80% = 1.25 for the scale:

   .. code-block:: shell-session

     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n'
     scale: 1.35484, angle: -37.8
     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n' --iter 2
     scale: 1.26448, angle: -27.6
     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n' --iter 4 --show
     scale: 1.24715, angle: -30

   So, four iterations are enough for a precise result!

#. And now something even more tricky - when a part of the image is cut out.
   The fourth and third samples are different just by the translation and by the fact that the feature we are matching against is incomplete on the fourth image.

   Generally, we have to address the cutoff.
   The ``--extend`` option here serves exactly this purpose.
   It extends the image by a given amount of pixels (on each side) and it tries to blur the cut-out image beyond its original border.
   Although the blurring might not look very impressive, it makes a huge difference for the image's spectrum which is used for the registration.
   So let's try:

   .. code-block:: shell-session

     [user@linuxbox examples]$ ird sample1.png sample4.png --extend 20 --show --print-result --iter 4
     scale: 0.745331
     angle: -36.580548
     shift: 165, 130

   As we can see, the result is correct.

   Extension can occur on-demand when the scale change or rotation operations result in image size growth.
   However, whether this will occur or not is not obvious, so it is advisable to specify the argument manually.
   In this example, specifying the option manually is not needed.

   .. warning::

     If the image extension by blurring is very different from how the image really looks like, the image registration will fail.

#. Buy what do we actually get on output?
   You may wonder what those numbers mean.
   The scale and angle information is quite clear, but the translation depends on the center of scaling and the center of rotation...
   So the idea is as follows.
   Let's assume you have an image, an ``imreg_dft`` output and all you want is to perform the image transformation yourself.
   The output describes what operations to perform on the image so it is close to the template.
   All transformations are performed using ``scipy.ndimage.interpolate`` package and you need to do the following:

   i. Call the ``zoom`` function with the provided scale.
      The center of the zoom is the center of the image.

   #. Then, rotate the image using the ``rotate`` function, specifyinh the angle you got on the output.
      The center of the rotation is again the center of the image.

   #. Finally, translate the image using the ``shift`` function.
      Remember that the ``y`` axis is the first one and ``x`` the second one.

   #. That's it, the image should now look like the template.

Advanced tweaking
+++++++++++++++++

There are some strange options you can use, we will explain their meaning now:

``--lowpass``, ``--highpass``: These two concern filtration of the image prior to the registration.
    There can be multiple reasons why to filter images:

    * One of them is filtered already due to conditions beyond your control, so by filtering them again just brings the other one on the par with the first one.

    * You want to filter out low frequencies since they are of no good when registering images anyway.

    The domain of the spectrum is a set of spatial frequencies.
    Each spatial frequency in an image is a vector with a :math:`x` and :math:`y` components.
    You can norm the frequencies by stating that the highest value of a compnent is 1, and denote value of spatial frequency as the (euclidean) length of the normed vector.
    Therefore the spatial frequencies of greatest values of :math:`\sqrt 2` are (1, 1), (1, -1) etc.

    An argument to a ``--lowpass`` or ``--highpass`` option is a tuple, usualy a number between 0 and 1.
    This number relates to the value of spatial frequencies it affects.
    For example, passing ``--lowpass 0.2,0.4`` means that spatial frequencies with value ranging from 0 to 0.2 will pass, whereas those with higher value than 0.4 won't.
    Spatial frequencies with values in between the two will be attenuated linearly.

``--filter-pcorr``: Fitering of phase correlation applies when determining the right translation vector.
    If the image pattern is not sampled very densely (i.e. close or even below the Nyquist frequency), ripples may appear near edges in the image.
    These ripples basically interfere with the algorithm and the phase correlation filtration may overcome this problem.

``--exponent``: When finding the right angle and scale, the highest element in an array is searched for.
    However, again due to incorrect sampling, it might not be the best guess --- for instance, this approach has the obvious flaw of being numerically unstable.
    There may be several extreme values close together and picking the center of them can be much better.
    This option plays the following role in the process:
    
    * The array is powered by the exponent.

    * The coordinates of the center of mass of the array are determined. 

    Formally: Let :math:`f(x)` be a discrete, 1-variable non-negative function, for instance :math:`f(0) = 3,\ f(1) = 0, f(2) = 2.99, f(3) = 1`.
    Then, the index of the greatest value is denoted by :math:`\mathrm{argmax}\, f(x) = 0`, because :math:`f(0)` is the greatest of :math:`f(x)` for all :math:`x` whete :math:`f(x)` is defined.
    The coordinate of the center of mass of :math:`f(x)` is :math:`t_f = \sum f(x_i)^c x_i / \sum f(x_i)^c`, where :math:`c` is our exponent, in case of real center of mass, :math:`c = 1`.
    The problem is that in this case, the value of :math:`\mathrm{argmax}\, f(x)` is unstable, since the difference between :math:`f(0)` and :math:`f(2)` is relatively low.
    If we consider real-world conditions, it could be below a fraction of the noise standard deviation.
    However, if we select a value of :math:`c = 5`, the value of :math:`t_f = 0.996`, which is just between the two highest values and not affected by :math:`f(3)`.
    And this is actually exactly what we want --- the interpolation during image transformations is not perfect and an analogous situation can occur in the spectrum and the center of few extreme values close together is more representative than the location of just one extreme value.

    One can generalize this to the case of 2D discrete functions and that's our case.
    Obviously, the higher the exponent is, the closer are we to picking the coordinates of the greatest array element.
    To neutralize the influence of points with low value, set the value of the exponent to a value greater or equal to 5.

    .. code-block:: shell-session

      [user@linuxbox ir_dft]$ ird resources/examples/sample1.png resources/examples/sample3.png --exponent inf --print-result
      scale: 1.357143
      angle: -37.800000
      shift: 37, 92
      [user@linuxbox ir_dft]$ ird resources/examples/sample1.png resources/examples/sample3.png --exponent 5 --print-result
      scale: 1.321192
      angle: -34.800000
      shift: 38, 84

    Although if we increase the number of iteration, the exponent won't make a difference.
    However, we can see that with only one iteration, setting the ``--exponent`` to ``5`` results in more precise result than the default value of ``'inf'``.
    The correct value is 1.25 for scale and -30 for angle.
