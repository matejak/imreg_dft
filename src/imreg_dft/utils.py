# -*- coding: utf-8 -*-
# utils.py

# Copyright (c) 2014-?, Matěj Týč
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
FFT based image registration. --- utility functions
"""

import numpy as np
import numpy.fft as fft


def _get_emslices(shape1, shape2):
    """
    Common code used by :func:`embed_to` and :func:`undo_embed`
    """
    slices_from = []
    slices_to = []
    for dim1, dim2 in zip(shape1, shape2):
        diff = dim2 - dim1
        # In fact: if diff == 0:
        slice_from = slice(None)
        slice_to = slice(None)

        # dim2 is bigger => we will skip some of their pixels
        if diff > 0:
            # diff // 2 + rem == diff
            rem = diff - (diff // 2)
            slice_from = slice(diff // 2, dim2 - rem)
        if diff < 0:
            diff *= -1
            rem = diff - (diff // 2)
            slice_to = slice(diff // 2, dim1 - rem)
        slices_from.append(slice_from)
        slices_to.append(slice_to)
    return slices_from, slices_to


def undo_embed(what, orig_shape):
    """
    Undo an embed operation

    :param what: What has once be the destination array
    :param what: The shape of the once original array

    :returns: The closest we got to the undo
    """
    _, slices_to = _get_emslices(what.shape, orig_shape)

    res = what[slices_to[0], slices_to[1]].copy()
    return res


def embed_to(where, what):
    """
    Given a source and destination arrays, put the source into
    the destination so it is centered and perform all necessary operations
    (cropping or aligning)

    :param where: The destination array (also modified inplace)
    :param what: The source array

    :returns: The destination array
    """
    slices_from, slices_to = _get_emslices(where.shape, what.shape)

    where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
    return where


def extend_by(what, dst):
    """
    Given a source array, extend it by given number of pixels and try
    to make the extension smooth (not altering the original array).
    """
    olddim = np.array(what.shape, dtype=int)
    newdim = olddim + 2 * dst

    bgval = get_borderval(what, dst)

    dest = np.zeros(newdim)
    res = dest.copy() + bgval
    res = embed_to(res, what)

    mask = dest
    mask = embed_to(mask, np.ones_like(what))

    res = frame_img(res, mask, dst)

    return res


def unextend_by(what, dst):
    """
    Try to undo as much as the :func:`extend_by` does.
    Some things can't be undone, though.
    """
    newdim = np.array(what.shape, dtype=int)
    origdim = newdim - 2 * dst

    res = undo_embed(what, origdim)
    return res


def imfilter(img, lows=None, highs=None):
    """
    Given an image, it applies a list of high-pass and low-pass filters on its
    Fourier spectrum.
    """
    if lows is None:
        lows = []
    if highs is None:
        highs = []

    dft = fft.fft2(img)
    for spec in highs:
        highpass(dft, spec[0], spec[1])
    for spec in lows:
        lowpass(dft, spec[0], spec[1])
    ret = np.real(fft.ifft2(dft))
    return ret


def highpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= (1 - mask)


def lowpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= mask


def _xpass(shape, lo, hi):
    """
    Computer a pass-filter mask with values ranging from 0 to 1.0
    The mask is low-pass, application has to be handled by a calling funcion.
    """
    assert lo <= hi, \
        "Filter order wrong, low '%g', high '%g'" % (lo, hi)
    assert lo >= 0, \
        "Low filter lower than zero (%g)" % lo
    # High can be as high as possible

    dom_x = np.fft.fftfreq(shape[0])[:, np.newaxis]
    dom_y = np.fft.fftfreq(shape[1])[np.newaxis, :]

    # freq goes 0..0.5, we want from 0..1, so we multiply it by 2.
    dom = np.sqrt(dom_x ** 2 + dom_y ** 2) * 2

    res = np.ones(dom.shape)
    res[dom >= hi] = 0.0
    mask = (dom > lo) * (dom < hi)
    res[mask] = 1 - (dom[mask] - lo) / (hi - lo)

    return res


def get_apofield(shape, aporad):
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, \
            "Apodization radius %d too big for shape dim. %d" % (aporad, dim)
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield


def frame_img(img, mask, dst):
    """
    Given an array, a mask (floats between 0 and 1), and a distance,
    alter the area where the mask is low (and roughly within dst from the edge)
    so it blends well with the area where the mask is high.
    The purpose of this is removal of spurious frequencies in the image's
    Fourier spectrum.

    Args:
        img (np.array): What we want to alter
        maski (np.array): The indicator what can be altered and what not
        dst (int): Parameter controlling behavior near edges, value could be
            probably deduced from the mask.
    """
    import scipy.ndimage as ndimg

    radius = dst / 1.8

    convmask0 = mask + 1e-8

    krad = radius * 2
    convimg = img
    convmask = convmask0
    convimg0 = img

    while krad > 0.2:
        convimg = ndimg.gaussian_filter(convimg0 * convmask0,
                                        krad, mode='wrap')
        convmask = ndimg.gaussian_filter(convmask0, krad, mode='wrap')
        convimg /= convmask
        convmask **= 0.5
        convimg = convimg * convmask + convimg0 * (1 - convmask)
        krad /= 1.6
        convimg0 = convimg

    convimg[mask == 1] = img[mask == 1]

    return convimg


def get_borderval(img, radius):
    """
    Given an image and a radius, examine the average value of the image
    at most radius pixels from the edge
    """
    mask = np.zeros_like(img, dtype=np.bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[radius, :] = True
    mask[-radius:, :] = True

    mean = np.median(img[mask])
    return mean
