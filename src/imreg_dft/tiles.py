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


_TILES = None
_SUCCS = None
_SHIFTS = None
_ANGLES = None
_SCALES = None
_DIFFS = None
_IMAGE = None
_OPTS = None
_POSS = None


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
        _SHIFTS[ii][0] = resdict["tvec"][0]
        _SHIFTS[ii][1] = resdict["tvec"][1]
        _SUCCS[ii] = resdict["success"]
    # Some tiles may have failed so much, that "angle" etc. are not defined
    # So we just mark them as failiures and move on.
    except KeyError:
        _SUCCS[ii] = 0


def _assemble_resdict(ii):
    ret = dict(
        angle=_ANGLES[ii],
        scale=_SCALES[ii],
        tvec=_SHIFTS[ii],
        success=_SUCCS[ii],
    )
    return ret


def _preprocess_extend(ims, extend, low, high, rcoef):
    ims = [utils.extend_by(img, extend) for img in ims]
    bigshape = np.array([img.shape for img in ims]).max(0)

    ims = filter_images(ims, low, high)
    if rcoef != 1:
        ims = [resample(img, rcoef) for img in ims]
        bigshape *= rcoef

    # Make the shape of images the same
    bgs = [np.zeros(bigshape) + utils.get_borderval(img, 5)
           for img in ims]
    ims = [utils.embed_to(bg, img)
           for bg, img in zip(bgs, ims)]
    return ims


def _postprocess_unextend(ims, im2, extend):
    ret = [utils.unextend_by(img, extend)
           for img in ims + [im2]]
    return ret


def process_images(ims, opts, tosa=None, get_unextended=False,
                   reports=None):
    # lazy import so no imports before run() is really called
    from imreg_dft import imreg

    rcoef = opts["resample"]
    ims = _preprocess_extend(ims, opts["extend"],
                             opts["low"], opts["high"], rcoef)
    if reports is not None:
        reports["processed-0"] = ims

    resdict = imreg._similarity(
        ims[0], ims[1], opts["iters"], opts["order"], opts["constraints"],
        opts["filter_pcorr"], opts["exponent"], reports=reports)

    if reports is not None:
        import pylab as pyl
        for ii, im in enumerate(reports["ims-filt"]):
            pyl.figure(); pyl.title("filtered"); pyl.imshow(im.real); pyl.colorbar()
            pyl.savefig("filt-%d.png" % ii)
        for ii, im in enumerate(reports["dfts-filt"]):
            pyl.figure(); pyl.title("log abs dfts"); pyl.imshow(np.log(np.abs(im))); pyl.colorbar()
            pyl.savefig("logabs-%d.png" % ii)
        for ii, im in enumerate(reports["logpolars"]):
            pyl.figure(); pyl.title("log abs log-ploar"); pyl.imshow(np.log(np.abs(im))); pyl.colorbar()
            pyl.savefig("logpolar-%d.png" % ii)

    # Seems that the reampling simply scales the translation
    resdict["Dt"] /= rcoef
    ty, tx = resdict["tvec"]
    resdict["tx"] = tx
    resdict["ty"] = ty
    resdict["imgs"] = ims
    tform = resdict

    if tosa is not None:
        tosa[:] = ird.transform_img_dict(tosa, tform)

    if get_unextended:
        im2 = imreg.transform_img_dict(ims[1], resdict, order=opts["order"])

        if rcoef != 1:
            ims = [resample(img, 1.0 / rcoef) for img in ims]
            im2 = resample(im2, 1.0 / rcoef)

        resdict["unextended"] = _postprocess_unextend(ims, im2, opts["extend"])

    # We need this for the transform above
    resdict["tvec"] /= rcoef

    return resdict


def process_tile(ii):
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES, _DIFFS
    tile = _TILES[ii]
    image = _IMAGE
    opts = _OPTS
    pos = _POSS[ii]
    try:
        # TODO: Add unittests that zero success result
        #   doesn't influence anything
        resdict = process_images((tile, image), opts, None)
        resdict['tvec'] += pos
        if np.isnan(_DIFFS[0]):
            _DIFFS[0] = resdict["Dangle"]
            _DIFFS[1] = resdict["Dscale"]
            _DIFFS[2] = resdict["Dt"]
    except ValueError:
        # probably incompatible images due to high scale change, so we
        # just add some harmless stuff here and proceed.
        resdict = dict(success=0)
    _distribute_resdict(resdict, ii)
    _SUCCS[ii] = resdict["success"]
    if 1:
        print("%d: succ: %g" % (ii, resdict["success"]))
        import pylab as pyl
        resdict["tvec"] -= pos
        tosa = ird.transform_img_dict(image, resdict, 0, opts["order"])
        tosa = utils.unextend_by(tosa, opts["extend"])
        # ird.imshow(tile, image, tosa, cmap=pyl.cm.gray)
        pyl.show()
    if 0:
        print(ii, _SUCCS[ii])
        import pylab as pyl
        pyl.figure(); pyl.imshow(tile)
        pyl.show()


def _fill_globals(tiles, poss, image, opts):
    ntiles = len(tiles)
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES, _DIFFS
    global _TILES, _IMAGE, _OPTS, _POSS
    _SUCCS = np.empty(ntiles, float) + np.nan
    _SHIFTS = np.empty((ntiles, 2), float) + np.nan
    _ANGLES = np.empty(ntiles, float) + np.nan
    _SCALES = np.empty(ntiles, float) + np.nan
    # Dangle, Dscale, Dt
    _DIFFS = np.empty(3, float) + np.nan

    _TILES = np.empty((ntiles,) + tiles[0].shape)
    for ii, tile in enumerate(tiles):
        _TILES[ii, :] = tile

    _IMAGE = np.zeros_like(image) + image
    _OPTS = opts
    _POSS = tuple((tuple(pos) for pos in poss))


def settle_tiles(imgs, tiledim, opts):
    global _SHIFTS

    tiles, poss = zip(* ird.utils.decompose(imgs[0], tiledim, 0.35))

    _fill_globals(tiles, poss, imgs[1], opts)

    for ii, pos in enumerate(poss):
        process_tile(ii)

    """
    if ncores == 0:  # no multiprocessing (to see errors)
        _seg_init(mesh, dims, counter)
        data = map(_get_prep, allranges)
    else:
        pool = mp.Pool(
            processes=ncores,
            initializer=_seg_init,
            initargs=(mesh, dims, counter),
        )
        res = pool.map_async(_get_prep, allranges)
        pool.close()

        while not res.ready():
            reporter.update(counter.value)
            time.sleep(sleeptime)
        assert res.successful(), \
            "Some exceptions have likely occured"

        data = res.get()
    """

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

    bgval = utils.get_borderval(imgs[1], 5)

    ims = _preprocess_extend(imgs, opts["extend"],
                             opts["low"], opts["high"], opts["resample"])
    im2 = ird.transform_img_dict(ims[1], resdict, bgval, opts["order"])

    # TODO: This is kinda dirty
    resdict["unextended"] = _postprocess_unextend(ims, im2, opts["extend"])
    resdict["Dangle"], resdict["Dscale"], resdict["Dt"] = _DIFFS

    return resdict
