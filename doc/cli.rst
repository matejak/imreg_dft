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
     shift: 19, -79

   .. warning::

     Remember that images have the zero coordinate (i.e. the origin) in their upper left corner!

#. You can have the results print in a defined way.
   First of all though, let's move to the examples directory:

   .. code-block:: shell-session

     [user@linuxbox ir_dft]$ cd resources/examples
     [user@linuxbox examples]$ ird sample1.png sample2.png --print-result --print-format 'translation:%(tx)d,%(ty)d\n'
     translation:19,-79


#. Let's try something tricky - the first and third example!

   .. code-block:: shell-session

     [user@linuxbox examples]$ ird sample1.png sample3.png --show

   Uh-oh, that didn't turn out very well, did it?
   The result is somehow better than the input, but highly unsatisfactory nevertheless.

   However, we have a triumph in our sleeve.
   We can force ``ir_dft`` to try to guess scale and rotation multiple times in a row.
   The correct values are -30° for the rotation and 80% for the scale:

   .. code-block:: shell-session

     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n'
     scale: 0.738889, angle: -37.8
     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n' --iter 2
     scale: 0.791667, angle: -27.6
     [user@linuxbox examples]$ ird sample1.png sample3.png --print-result --print-format 'scale: %(scale)g, angle: %(angle)g\n' --iter 4 --show
     scale: 0.802817, angle: -30

   So, four iterations are enough for a precise result!

#. Buy what do we actually get?
   However, you may wonder what those numbers mean.
   The scale and angle information is clear, but the translation depends on the center of scaling and the center of rotation...
   So the idea is as follows.
   Let's assume you have an image, an ``imreg_dft`` output and all you want is to perform the image transformation yourself.
   All transformations are performed using ``scipy.ndimage.interpolate`` package and you need to do the following:

   i. Call the ``zoom`` function with the inverse of the scale.
      The center of the zoom is the center of the image.

   #. Then, rotate the image using the ``rotate`` function, using the negative of the angle you got on the output.
      The center of the rotation is again the center of the image.

   #. Finally, translate the image using the ``shift`` function.
      Remember that the ``y`` axis is the first one and ``x`` the second one.

