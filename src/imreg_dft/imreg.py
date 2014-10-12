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
from numpy.fft import fft2, ifft2, fftshift

import imreg_dft.utils as utils

try:
    import scipy.ndimage as ndi
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage as ndi
    import ndimage.interpolation as ndii

__docformat__ = 'restructuredtext en'
__all__ = ['translation', 'similarity']


def similarity(im0, im1, order, filter_pcorr):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2-dimensional.")

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    bgval = utils.get_borderval(im1, 5)
    im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

    # now we can use pcorr to guess the translation
    t0, t1 = translation(im0, im2, filter_pcorr)

    im2 = transform_img(im2, 1, 0, t0, t1, bgval=bgval, order=order)
    im2 = transform_img(im1, scale, angle, t0, t1, bgval, order=order)

    imask = transform_img(np.ones_like(im1), scale, angle, t0, t1,
                          0, order=order)

    im2 = utils.frame_img(im2, imask, 20)

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        t0, t1 = d + t1, d + t0

    # don't know what it does, but it alters the scale a little bit
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    return im2, scale, angle, [-t0, -t1]


def translation(im1, im2, filter_pcorr=0):
    """Return translation vector to register images."""
    f0 = fft2(im1)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    if filter_pcorr > 0:
        ir = ndi.minimum_filter(ir, filter_pcorr)

    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    return t0, t1


def transform_img(src, scale=1.0, angle=0.0, t0=0, t1=0, bgval=0, order=1):
    dest0 = src.copy()
    if scale != 1.0:
        dest0 = ndii.zoom(dest0, 1.0 / scale, order=order, cval=bgval)
    if angle != 0.0:
        dest0 = ndii.rotate(dest0, angle, order=order, cval=bgval)

    if t0 != 0 or t1 != 0:
        dest0 = ndii.shift(dest0, [t0, t1], order=order, cval=bgval)

    bg = np.zeros_like(src) + bgval
    dest = utils.embed_to(bg, dest0)

    return dest


def similarity_matrix(scale, angle, vector):
    """Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    S = np.diag([scale, scale, 1.0])
    R = np.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = np.identity(3)
    T[:2, 2] = vector
    return np.dot(T, np.dot(R, S))


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    d = np.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii,
                                             dtype=np.float64)) - 1.0
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = np.outer(
        np.cos(np.linspace(- math.pi / 2., math.pi / 2., shape[0])),
        np.cos(np.linspace(- math.pi / 2., math.pi / 2., shape[1])))
    return (1.0 - x) * (2.0 - x)


def imread(fname, norm=True):
    """Return image data from img&hdr uint8 files."""
    with open(fname + '.hdr', 'r') as fh:
        hdr = fh.readlines()
    img = np.fromfile(fname + '.img', np.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype(np.float64)
        img /= 255.0
    return img


def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot

    if cmap is None:
        cmap = 'coolwarm'
    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.grid()
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.grid()
    pyplot.show()
