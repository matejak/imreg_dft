# -*- coding: utf-8 -*-
# tiles.py

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


import numpy as np

import imreg_dft as ird
import imreg_dft.utils as utils


_SUCCS = None
_SHIFTS = None
_ANGLES = None
_SCALES = None


def resample(img, coef):
    from scipy import signal
    ret = img
    for axis in range(2):
        newdim = ret.shape[axis] * coef
        ret = signal.resample(ret, newdim, axis=axis)
    return ret


def filter_images(imgs, low, high):
    ret = [utils.imfilter(img, low, high) for img in imgs]
    return ret


def _distribute_resdict(resdict, ii):
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES

    try:
        _ANGLES[ii] = resdict["angle"]
        _SCALES[ii] = resdict["scale"]
        _SHIFTS[ii][0] = resdict["ty"]
        _SHIFTS[ii][1] = resdict["tx"]
        _SUCCS[ii] = resdict["success"]
    # Some tiles may have failed so much, that "angle" etc. are not even defined
    # So we just mark them as failiures and move on.
    except KeyError:
        _SUCCS[ii] = 0


def _assemble_resdict(ii):
    ret = dict(
        angle=_ANGLES[ii],
        scale=_SCALES[ii],
        ty=_SHIFTS[ii][0],
        tx=_SHIFTS[ii][1],
        success=_SUCCS[ii],
    )
    return ret


def process_images2(ims, opts, tosa=None):
    pass


def process_images(ims, opts, tosa=None):
    # lazy import so no imports before run() is really called
    import numpy as np
    from imreg_dft import utils
    from imreg_dft import imreg

    ims = [utils.extend_by(img, opts["extend"]) for img in ims]
    bigshape = np.array([img.shape for img in ims]).max(0)

    ims = filter_images(ims, opts["low"], opts["high"])
    rcoef = opts["resample"]
    if rcoef != 1:
        ims = [resample(img, rcoef) for img in ims]
        bigshape *= rcoef

    # Make the shape of images the same
    ims = [utils.embed_to(np.zeros(bigshape) + utils.get_borderval(img, 5), img)
           for img in ims]

    resdict = imreg.similarity(
        ims[0], ims[1], opts["iters"], opts["order"], opts["constraints"],
        opts["filter_pcorr"], opts["exponent"])

    im2 = resdict.pop("timg")

    # Seems that the reampling simply scales the translation
    resdict["tvec"] /= rcoef
    ty, tx = resdict["tvec"]
    resdict["tx"] = tx
    resdict["ty"] = ty
    resdict["imgs"] = ims
    tform = resdict

    if tosa is not None:
        tosa[:] = ird.transform_img_dict(tosa, tform)

    if rcoef != 1:
        ims = [resample(img, 1.0 / rcoef) for img in ims]
        im2 = resample(im2, 1.0 / rcoef)
        resdict["Dt"] /= rcoef

    resdict["unextended"] = [utils.unextend_by(img, opts["extend"])
                             for img in ims + [im2]]

    return resdict


def process_tile(tile, imgs, opts, ii, pos):
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES
    try:
        # TODO: Add unittests that zero success result
        #   doesn't influence anything
        resdict = process_images((tile, imgs[1]), opts, None)
    except ValueError:
        # probably incompatible images due to high scale change, so we
        # just add some harmless stuff here and proceed.
        resdict = dict(success=0)
    _distribute_resdict(resdict, ii)
    if 0:
        print("%d: succ: %g" % (ii, resdict["success"]))
        import pylab as pyl
        _, _, tosa = resdict["unextended"]
        ird.imshow(tile, imgs[1], tosa, cmap=pyl.cm.gray)
        pyl.show()
    _SUCCS[ii] = resdict["success"]
    if 0:
        print(ii, _SUCCS[ii])
        import pylab as pyl
        pyl.figure(); pyl.imshow(tile)
        pyl.show()


def settle_tiles(tiles, imgs, tiledim, opts):
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES
    _SUCCS = np.empty(len(tiles), float) + np.nan
    _SHIFTS = np.empty((len(tiles), 2), float) + np.nan
    _ANGLES = np.empty(len(tiles), float) + np.nan
    _SCALES = np.empty(len(tiles), float) + np.nan

    tiles = ird.utils.decompose(imgs[0], tiledim, 0.35)
    for ii, (tile, pos) in enumerate(tiles):
        process_tile(tile, imgs, opts, ii, pos)

    tosa_offset = np.array(imgs[0].shape)[:2] - np.array(tiledim)[:2] + 0.5
    _SHIFTS -= tosa_offset / 2.0

    # Get the cluster of the tiles that have similar results and that look
    # most promising along with the index of the best tile
    cluster, amax = utils.get_best_cluster(_SHIFTS, _SUCCS, 5)
    # Make the quantities estimation even more precise by taking
    # the average of all good tiles
    shift, angle, scale, score = utils.get_values(
        cluster, _SHIFTS, _SUCCS, _ANGLES, _SCALES)

    resdict = _assemble_resdict(amax)
    resdict["scale"] = scale
    resdict["angle"] = angle
    resdict["tvec"] = shift
    resdict["ty"], resdict["tx"] = resdict["tvec"]

    orig = tiles[amax][0]
    bgval = utils.get_borderval(orig, 5)
    im2 = ird.transform_img_dict(orig, resdict, bgval, opts["order"])

    # TODO: This is kinda dirty
    resdict["unextended"] = [utils.unextend_by(img, opts["extend"])
                             for img in (imgs[1], orig, im2)]
    resdict["Dangle"], resdict["Dscale"] = ird.imreg._get_precision(img.shape,
                                                                    scale)
    resdict["Dt"] = 0.25

    return resdict
