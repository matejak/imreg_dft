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


def undo_embed(what, orig):
    _, slices_to = _get_emslices(what.shape, orig.shape)

    res = what[slices_to[0], slices_to[1]].copy()
    return res


def embed_to(where, what):
    slices_from, slices_to = _get_emslices(where.shape, what.shape)

    where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
    return where


def extend_by(what, dst):
    olddim = np.array(what.shape, dtype=int)
    newdim = olddim + dst

    bgval = get_borderval(what, dst)

    dest = np.zeros(newdim)
    res = dest.copy() + bgval
    embed_to(res, what)

    mask = dest
    embed_to(mask, np.ones_like(what))

    res = frame_img(res, mask, dst)

    return res


def unextend_by(what, dst):
    newdim = np.array(what.shape, dtype=int)
    origdim = newdim - dst

    res = undo_embed(what, np.empty(origdim))
    return res


def filter(img, lows=None, highs=None):
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
        assert dim > aporad * 2
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad - 1:] = apos[-aporad - 1:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield


def frame_img(img, mask, dst):
    import scipy.ndimage as ndimg

    radius = dst // 3

    mask += 1e-5
    convimg = ndimg.gaussian_filter(img * mask, radius, mode='wrap')
    convmask = ndimg.gaussian_filter(mask, radius, mode='wrap')

    wconvimg = convimg / convmask

    compmask = convmask - 0.5
    compmask[compmask < 0] = 0
    compmask /= compmask.max()

    apofield = get_apofield(img.shape, dst // 2)
    apoarr = compmask * apofield

    res = wconvimg * (1 - apoarr) + apoarr * img
    return res


def get_borderval(img, radius):
    mask = np.zeros_like(img, dtype=np.bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[radius, :] = True
    mask[-radius:, :] = True

    mean = img[mask].mean()
    return mean
