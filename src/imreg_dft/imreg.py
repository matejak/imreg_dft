# -*- coding: utf-8 -*-
# imreg.py

# Copyright (c) 2014-?, Matěj Týč
# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
FFT based image registration. --- main functions
"""

from __future__ import division, print_function

import math

import numpy as np
try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import numpy.fft as fft
import scipy.ndimage.interpolation as ndii

import imreg_dft.utils as utils


__all__ = ['translation', 'similarity', 'transform_img',
           'transform_img_dict', 'imshow']


def _logpolar_filter(adft):
    shape = adft.shape
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    filt = 1.0 - np.cos(np.sqrt(yy ** 2 + xx ** 2))**2
    ret = adft * filt
    return ret


def _get_pcorr_shape(shape):
    ret = (int(max(shape) * 1.0),) * 2
    return ret


def _get_ang_scale(ims, bgval, exponent='inf', constraints=None):
    """
    Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        bgval: We also pad here in the :func:`map_coordinates`
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`

    Returns:
        tuple: Scale, angle. Describes the relationship of the subject  image to
        the first one.
    """
    assert len(ims) == 2, \
        "Only two images are supported as input"
    shape = ims[0].shape

    adfts = [fft.fftshift(abs(fft.fft2(im))) for im in ims]
    adfts = [_logpolar_filter(adft) for adft in adfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(adft, pcorr_shape, log_base, 0.0) for adft in adfts]

    if 0:
        import pylab as pyl
        pyl.figure(); pyl.imshow(ims[0]);
        pyl.figure(); pyl.imshow(ims[1]);
        pyl.show()

    (arg_ang, arg_rad), success = _phase_correlation(
        stuffs[0], stuffs[1],
        utils.argmax_angscale, log_base, exponent, constraints)

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = utils.wrap_angle(angle, 360)
    scale = log_base ** arg_rad

    if not 0.5 < scale < 2:
        raise ValueError(
            "Images are not compatible. Scale change %g too big to be true."
            % scale)

    return 1.0 / scale, - angle


def _translation(im0, im2, filter_pcorr, odds=1, constraints=None):
    """
    Args:
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
    """
    angle = 0
    tvec, succ = translation(im0, im2, filter_pcorr, constraints)
    tvec2, succ2 = translation(im0, utils.rot180(im2),
                               filter_pcorr, constraints)
    if succ2 * odds > succ or odds == -1:
        tvec = tvec2
        succ = succ2
        angle += 180

    if 0:
        import pylab as pyl
        pyl.figure(); pyl.imshow(im0, cmap=pyl.cm.gray)
        pyl.show()
    return tvec, succ, angle


def _get_precision(shape, scale=1):
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.5
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.5
    return Dangle, Dscale


def similarity(im0, im1, numiter=1, order=3, constraints=None,
               filter_pcorr=0, exponent='inf'):
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        numiter (int): How many times to iterate when determining scale and
            rotation
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc.
        filter_pcorr (int): Radius of a spectrum filter for translation
            detection
        exponent (float or 'inf'): The exponent value used during processing.
            Refer to the docs for a thorough explanation. Generally, pass "inf"
            when feeling conservative. Otherwise, experiment, values below 5
            are not even supposed to work.
        constraints (dict or None): Specify preference of seeked values.
            Pass None (default) for no constraints, otherwise pass a dict with
            keys ``angle``, ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass
            all, some of them or none of them, all is fine). The value of a key
            is supposed to be a mutable 2-tuple (e.g. a list), where the first
            value is related to the constraint center and the second one to
            softness of the constraint (the higher is the number,
            the more soft a constraint is).

            More specifically, constraints may be regarded as weights
            in form of a shifted Gaussian curve.
            However, for precise meaning of keys and values,
            see the documentation section :ref:`constraints`.
            Names of dictionary keys map to names of command-line arguments.

    Returns:
        dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
        ``success`` and ``timg`` (the transformed subject image)

    .. note:: There are limitations

        * Scale change must be less than 2.
        * No subpixel precision (but you can use *resampling* to get
          around this).
    """
    shape = im0.shape
    if shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    constraints_default = dict(angle=[0, None], scale=[1, None])
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default.update(constraints)
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    bgval = utils.get_borderval(im1, 5)
    for ii in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], bgval, exponent,
                                            constraints_dynamic)
        scale *= newscale
        angle += newangle

        constraints_dynamic["scale"][0] /= newscale
        constraints_dynamic["angle"][0] -= newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

    # Here we look how is the turn-180
    target, stdev = constraints.get("angle", (0, None))
    odds = _get_odds(angle, target, stdev)

    # now we can use pcorr to guess the translation
    tvec, succ, angle2 = _translation(im0, im2, filter_pcorr, odds, constraints)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += angle2
    angle = utils.wrap_angle(angle, 360)

    # don't know what it does, but it alters the scale a little bit
    # scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    Dangle, Dscale = _get_precision(shape, scale)

    res = dict(
        scale=scale,
        angle=angle,
        tvec=tvec,
        Dscale=Dscale,
        Dangle=Dangle,
        Dt=0.5,
        success=succ
    )

    im2 = transform_img_dict(im1, res, bgval, order)
    # Order of mask should be always 1 - higher values produce strange results.
    imask = transform_img_dict(np.ones_like(im1), res, 0, 1)
    # This removes some weird artifacts
    imask[imask > 0.8] = 1.0

    # Framing here = just blending the im2 with its BG according to the mask
    im2 = utils.frame_img(im2, imask, 10)

    res["timg"] = im2
    return res


def _get_odds(angle, target, stdev):
    """
    Args:

    Return:
        float: The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
    """
    ret = 1
    if stdev is not None:
        diffs = [abs(utils.wrap_angle(ang, 360))
                 for ang in (target - angle, target - angle + 180)]
        odds0, odds1 = 0, 0
        if stdev > 0:
            odds0, odds1 = [np.exp(- diff ** 2 / stdev ** 2) for diff in diffs]
        if odds0 == 0 and odds1 > 0:
            # -1 is treated as infinity in _translation
            ret = -1
        elif stdev == 0 or (odds0 == 0 and odds1 == 0):
            ret = -1
            if diffs[0] < diffs[1]:
                ret = 0
        else:
            ret = odds1 / odds0
    return ret


def translation(im0, im1, filter_pcorr=0, constraints=None):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of a spectrum filter for translation
            detection
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.

    Returns:
        tuple: The translation vector and success number: ((Y, X), success)
    """
    ret, succ = _phase_correlation(
        im0, im1,
        utils.argmax_translation, filter_pcorr, constraints)
    return ret, succ


def _phase_correlation(im0, im1, callback=None, * args):
    """
    Computes phase correlation between im0 and im1

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X)
    """
    if callback is None:
        callback = utils.argmax2D
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered, so we take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im2
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    if 0:
        import pylab as pyl
        pyl.figure(); pyl.imshow(im0);
        pyl.figure(); pyl.imshow(im1);
        pyl.figure(); pyl.imshow(scps);
        pyl.show()

    (t0, t1), success = callback(scps, * args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def transform_img_dict(img, tdict, bgval=None, order=1, invert=False):
    """
    Wrapper of :func:`transform_img`, works well with the :func:`similarity`
    output.

    Args:
        img
        tdict (dictionary): Transformation dictionary --- supposed to contain
            keys "scale", "angle" and "tvec"
        bgval
        order
        invert (bool): Whether to perform inverse transformation --- doesn't
            work very well with the translation.

    Returns:
        :The same as :func:`transform_img`
    """
    scale = tdict["scale"]
    angle = tdict["angle"]
    tvec = np.array(tdict["tvec"])
    if invert:
        scale = 1.0 / scale
        angle *= -1
        tvec *= -1
    res = transform_img(img, scale, angle, tvec, bgval=bgval, order=order)
    return res


def transform_img(img, scale=1.0, angle=0.0, tvec=(0, 0), bgval=None, order=1):
    """
    Return translation vector to register images.

    Args:
        img (2D or 3D numpy array): What will be transformed.
            If a 3D array is passed, it is treated in a manner in which RGB
            images are supposed to be handled - i.e. assume that coordinates
            are (Y, X, channels).
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc. Linear works surprisingly well.

    Returns:
        The transformed img, may have another i.e. (bigger) shape than
            the source.
    """
    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], scale, angle, tvec,
                                     bgval, order)
        return ret

    if bgval is None:
        bgval = utils.get_borderval(img, 5)

    bigshape = np.array(img.shape) * 1.2
    bg = np.zeros(bigshape, img.dtype) + bgval

    dest0 = utils.embed_to(bg, img.copy())
    if scale != 1.0:
        dest0 = ndii.zoom(dest0, scale, order=order, cval=bgval)
    if angle != 0.0:
        dest0 = ndii.rotate(dest0, angle, order=order, cval=bgval)

    if tvec[0] != 0 or tvec[1] != 0:
        dest0 = ndii.shift(dest0, tvec, order=order, cval=bgval)

    bg = np.zeros_like(img) + bgval
    dest = utils.embed_to(bg, dest0)
    return dest


def similarity_matrix(scale, angle, vector):
    """
    Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    raise NotImplementedError("We have no idea what this is supposed to do")
    m_scale = np.diag([scale, scale, 1.0])
    m_rot = np.identity(3)
    angle = math.radians(angle)
    m_rot[0, 0] = math.cos(angle)
    m_rot[1, 1] = math.cos(angle)
    m_rot[0, 1] = -math.sin(angle)
    m_rot[1, 0] = math.sin(angle)
    m_transl = np.identity(3)
    m_transl[:2, 2] = vector
    return np.dot(m_transl, np.dot(m_rot, m_scale))


def _get_log_base(shape, new_r):
    """
    Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.
    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * 1.1
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    # the base is chosen so that 'new_r' = log_'base'('old_r')
    log_base = np.exp(np.log(old_r) / (new_r))
    return log_base


def _logpolar(image, shape, log_base, bgval=None):
    """
    Return log-polar transformed image and log base.
    Takes into account anisotropicity of the freq spectrum of rectangular images

    Args:
        image: The image to be transformed
        shape: Shape of the transformed image
        log_base: Parameter of the transformation, convoluted with
            :func:`_get_log_base`

    Returns:
        The transformed image
    """
    if bgval is None:
        bgval = utils.get_borderval(image, 5)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_scales(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    ndii.map_coordinates(image, [y, x], output=output, order=3,
                         mode="constant", cval=bgval)
    """
    import pylab as pyl
    pyl.figure(); pyl.imshow(output);
    pyl.show()
    """
    return output


def imshow(im0, im1, im2, cmap=None, fig=None, **kwargs):
    """
    Plot images using matplotlib.
    Opens a new figure with four subplots:

    ::

      +----------------------+---------------------+
      |                      |                     |
      |   <template image>   |   <subject image>   |
      |                      |                     |
      +----------------------+---------------------+
      | <difference between  |                     |
      |  template and the    |<transformed subject>|
      | transformed subject> |                     |
      +----------------------+---------------------+

    Args:
        im0 (np.ndarray): The template image
        im1 (np.ndarray): The subject image
        im2: The transformed subject --- it is supposed to match the template
        cmap (optional): colormap
        fig (optional): The figure you would like to have this plotted on

    Returns:
        matplotlib figure: The figure with subplots
    """
    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.figure()
    if cmap is None:
        cmap = 'coolwarm'
    # We do the difference between the template and the result now
    # To increase the contrast of the difference, we norm images according
    # to their near-maximums
    norm = np.percentile(im2, 99.5) / np.percentile(im0, 99.5)
    im3 = abs(im2 - im0 * norm)
    pl0 = fig.add_subplot(221)
    pl0.imshow(im0, cmap, **kwargs)
    pl0.grid()
    share = dict(sharex=pl0, sharey=pl0)
    pl = fig.add_subplot(222, ** share)
    pl.imshow(im1, cmap, **kwargs)
    pl.grid()
    pl = fig.add_subplot(223, ** share)
    pl.imshow(im3, cmap, **kwargs)
    pl.grid()
    pl = fig.add_subplot(224, ** share)
    pl.imshow(im2, cmap, **kwargs)
    pl.grid()
    return fig
