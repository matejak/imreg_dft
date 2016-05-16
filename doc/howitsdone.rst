Conceptual-level documentation
==============================

Image registration procedure
----------------------------

Now, let's examine the :func:`imreg_dft.imreg.similarity` function.
It estimates the scale, rotation and translation relationship between the images and we will take a look what are the means by which the end-result is obtained.

The image registration is carried out as follows:

#. Images (template and subject) are loaded in a form of 2D or 3D numpy arrays, where coordinates have this meaning: ``(y, x [, channel])`` .
#. Both images are filtered.
   Typically, low spatial frequencies are stripped, because they are not useful for the phase correlation.
   This is done in :func:`imreg_dft.utils.imfilter`

#. If requested, both images are resampled (in other words, upscaled).
#. If necessary, both images are extended so that their shapes match.
   Implementation in :func:`imreg_dft.utils.embed_to`, gets called by :func:`imreg_dft.tiles._preprocess_extend`, 

#. Phase correlation is performed to determine angle---scale change (:func:`imreg_dft.imreg.similarity`):

   a. Images are apodized (so they are seamless with respect of their borders) in :func:`imreg_dft.imreg._get_ang_scale` 
      by calling :func:`imreg_dft.utils._apodize`.
   #. Amplitude of the Fourier spectrum is calculated and the log-polar transformation is performed.
   #. Phase correlation is performed on that log-polar spectrum amplitude.
   #. Source image is transformed to match the template (according to the angle and scale change).

#. Second round of phase correlation is performed to determine translation (:func:`imreg_dft.imreg.translation`).
   Images are already apodized and compatible (this is ensured in the previous step).

   a. Phase correlation on spectrums of the template and the transformed subject is performed. 
   #. Phase correlation on spectrums of the template and the transformed subject rotated over 180° is performed.
   #. Results of both operations are compared and the one that is more successful serves as final determination of angle and true translation vector.
      This is due to the fact that the determination of angle is ambiguous.

#. The result (transformation parameters, transformed subject, ...) is saved to a dictionary.
#. If a transformed subject is requested (e.g. if you want to compare it with the template pixel-by-pixel), it is made (by undoing extending and resampling operations).

Translation
+++++++++++

The phase correlation method is able to guess translation from the phase of image's spectrum (i.e. its Fourier transform).
For more in-depth reading consult the `Wikipedia entry <https://en.wikipedia.org/wiki/Phase_correlation>`_.
The short-hand explanation is that translation of function is possible by taking its spectrum, multiplying it by a complex function and inverting it back to image.
Hence, when we have two shifted images, it is obviously possible to guess their translation from their spectra.

The image is an array of real numbers, therefore its spectrum `is symmetric in a way <https://en.wikipedia.org/wiki/Hermitian_function>`_.
This is the reason why the translation is checked first of all on the two images, and then one of them is rotated 180 degrees and the check is repeated.

Performing phase correlation on the two images means:

* Spectra are calculated from respective images.
* Cross-power spectrum is calculated:

  .. math::

    R = \frac{F_1 \bar F_2} {|F_1| |F_2| + \varepsilon}

  where :math:`F_{1, 2}` are Fourier transforms (i.e. spectra) of input images (:math:`\bar F_2` is a complex conjugate of :math:`F_2`) and :math:`\varepsilon` is a very small positive real number.
  Note that it is normalized, so :math:`\max R = 1` (when not taking :math:`\varepsilon` into account).

* The input for phase correlation is calculated:

  .. math::

    R_i = |F^{-1}(R)| ,

  where :math:`R` is the cross-power spectrum and :math:`F^{-1}` is the inverse Fourier transform operator.
     
* The `shifted <http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.fft.fftshift.html>`_ cross-power spectrum is passed to :func:`imreg_dft.utils.argmax_translation` and translation vector and success value are returned.

  There are arguments passed to the translation estimate function:

  * ``filter_pcorr``: Radius of a minimum filter.
    Typically, when images are just translated, a translation one pixel off is still quite good.
    The phase correlation method heavily relies on image's high frequencies and sometimes there may be one image translation that looks good from the phase correlation perspective.
    If we apply a `minimum filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html#scipy.ndimage.minimum_filter>`_, those false positives disappear, whereas the true result is affected much less.

  * ``constraints``: Sometimes, we roughly know how the translation should be.
    Therefore, we can specify it, and it will be less likely that it will pick solution that is more favorable, but differs from the constraint.

  * ``report``: When something goes wrong, it is good to have some insight into how internal data inside of the function looked like.

* The function outputs the translation vector and a success value --- the value of ... (to be continued)

Rotation and scale
++++++++++++++++++

The front-end
-------------

