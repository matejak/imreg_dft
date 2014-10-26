Command-line tool overview
==========================

The package contains one Python :abbr:`CLI (command-line interface)` script.
Although you are more likely to use the ``imreg_dft`` functionality from your own Python programs, you can still have some use to the ``ird`` front-end.
There are three main reasons why you would want to use it:

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
