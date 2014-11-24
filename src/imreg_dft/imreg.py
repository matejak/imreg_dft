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
import numpy.fft as fft
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as ndii

import imreg_dft.utils as utils


__all__ = ['translation', 'similarity', 'transform_img',
           'transform_img_dict', 'imshow']

EXPO = 'inf'


def _get_ang_scale(ims, exponent=EXPO):
    """
    Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`

    Returns:
        tuple: Scale, angle. Describes the relationship of the second image to
        the first one.
    """
    assert len(ims) == 2, \
        "Only two images are supported as input"
    shape = ims[0].shape

    adfts = [fft.fftshift(abs(fft.fft2(im))) for im in ims]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    stuffs = [_logpolar(adft, shape[1]) for adft in adfts]
    log_base = _get_log_base(shape, shape[1])

    stuffs = [fft.fft2(stuff) for stuff in stuffs]
    r0 = abs(stuffs[0]) * abs(stuffs[1])
    ir = abs(fft.ifft2((stuffs[0] * stuffs[1].conjugate()) / r0))

    i0, i1 = utils._argmax_ext(ir, exponent)

    angle = -180.0 * i0 / ir.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        ir = abs(fft.ifft2((stuffs[1] * stuffs[0].conjugate()) / r0))
        i0, i1 = utils._argmax_ext(ir, exponent)
        angle = 180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    return 1.0 / scale, - angle


def similarity(im0, im1, numiter=1, order=3, filter_pcorr=0, exponent=EXPO):
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second image
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

    Returns:
        dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X)
        and ``timg`` (the transformed image)

    .. note:: There are limitations

        * Scale change must be less than 1.8.
        * No subpixel precision (but you can use *resampling* to get
          around this).
    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    bgval = utils.get_borderval(im1, 5)
    for ii in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], exponent)
        scale *= newscale
        angle += newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

    # now we can use pcorr to guess the translation
    tvec = translation(im0, im2, filter_pcorr)

    # don't know what it does, but it alters the scale a little bit
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    res = dict(
        scale=scale,
        angle=angle,
        tvec=tvec,
    )

    im2 = transform_img_dict(im1, res, bgval, order)
    imask = transform_img_dict(np.ones_like(im1), res, 0, order)
    # for some reason, when using cubic interp, the mask becomes quite strange
    # and low.
    imask[imask > 0.8] = 1.0

    # Framing here = just blending the im2 with its BG according to the mask
    im2 = utils.frame_img(im2, imask, 10)

    res["timg"] = im2
    return res

    # correct parameters for ndimage's internal processing
    # Probably calculated for the case the tform center is 0, 0 (vs center of
    # the image)
    if angle > 0.0:
        dif = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        tvec = dif + tvec[1], dif + tvec[0]
    elif angle < 0.0:
        dif = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        tvec = dif + tvec[1], dif + tvec[0]

    # We don't know what this is supposed to do, so it is here.


def translation(im0, im1, filter_pcorr=0):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second image
        filter_pcorr (int): Radius of a spectrum filter for translation
            detection
    Returns:
        tuple: The translation vector (Y, X)
    """
    f0 = fft.fft2(im0)
    f1 = fft.fft2(im1)
    # spectrum can be filtered, so we take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    ir = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    if filter_pcorr > 0:
        ir = ndi.minimum_filter(ir, filter_pcorr)

    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    return np.array((t0, t1), dtype=float)


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
        img (2D numpy array): What will be transformed
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc.

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
    dest0 = img.copy()
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


def _get_log_base(shape, radii):
    """
    Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`
    """
    center = shape[0] / 2, shape[1] / 2
    d = np.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    return log_base


def _logpolar(image, radii, angles=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    log_base = _get_log_base(shape, radii)
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii,
                                             dtype=np.float64)) - 1.0
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output


def imshow(im0, im1, im2, cmap=None, fig=None, **kwargs):
    """
    Plot images using matplotlib.
    Opens a new figure with four subplots:

    ::

      +---------------------+---------------------+
      |                     |                     |
      |   <template image>  |       <image>       |
      |                     |                     |
      +---------------------+---------------------+
      | <difference between |                     |
      |   template and the  | <transformed image> |
      |  transformed image> |                     |
      +---------------------+---------------------+

    Args:
        im0 (np.ndarray): The template image
        im1: The ``image``
        im2: The transformed ``image`` --- it is supposed to match the template
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
    norm = np.percentile(im2, 95) / np.percentile(im0, 95)
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
