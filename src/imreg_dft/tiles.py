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
import imreg_dft.reporting as reporting


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


def filter_images(imgs, low, high, cut):
    ret = [utils.imfilter(img, low, high, cut) for img in imgs]
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


def _preprocess_extend(ims, extend, low, high, cut, rcoef):
    bigshape = np.array([img.shape for img in ims]).max(0) + 2 * extend
    bigshape *= rcoef
    ims = [_preprocess_extend_single(im, extend, low, high, cut,
                                     rcoef, bigshape)
           for im in ims]

    # Safeguard that the earlier determination of bigshape was correct.
    assert np.all(bigshape == np.array([img.shape for img in ims]).max(0))
    return ims


def _preprocess_extend_single(im, extend, low, high, cut, rcoef, bigshape):
    im = utils.extend_by(im, extend)
    im = utils.imfilter(im, low, high, cut)
    if rcoef != 1:
        im = resample(im, rcoef)

    # Make the shape of images the same
    bg = np.zeros(bigshape) + utils.get_borderval(im, 5)
    im = utils.embed_to(bg, im)
    return im


def _postprocess_unextend(ims, im2, extend, rcoef=1):
    if rcoef != 1:
        ims = [resample(img, 1.0 / rcoef) for img in ims]
        im2 = resample(im2, 1.0 / rcoef)

    ret = [utils.unextend_by(img, extend)
           for img in ims + [im2]]
    return ret


def _savefig(fig, fname):
    fig.savefig(fname, bbox_inches="tight")
    fig.clear()


def process_images(ims, opts, tosa=None, get_unextended=False,
                   reports=None):
    """
    Args:
        tosa (np.ndarray): An array where to save the transformed subject.
        get_unextended (bool): Whether to get the transformed subject
            in the same shape and coord origin as the template.
    """
    # lazy import so no imports before run() is really called
    from imreg_dft import imreg

    rcoef = opts["resample"]
    ims = _preprocess_extend(ims, opts["extend"],
                             opts["low"], opts["high"], opts["cut"], rcoef)
    if reports is not None:
        reports["processed-0"] = ims

    resdict = imreg._similarity(
        ims[0], ims[1], opts["iters"], opts["order"], opts["constraints"],
        opts["filter_pcorr"], opts["exponent"], reports=reports)

    if reports is not None:
        import pylab as pyl
        fig = pyl.figure(figsize=(18, 6))
        prefix = "report"
        for key, value in reports.items():
            if "ims-filt" in key:
                reporting.imshow_plain(fig, value,
                                       ("template", "sample"), True)

                fname = "%s-%s.png" % (prefix, key)
                _savefig(fig, fname)
            elif "dfts-filt" in key:
                reporting.imshow_spectra(fig, value)

                fname = "%s-%s.png" % (prefix, key)
                _savefig(fig, fname)
            elif "logpolars" in key:
                reporting.imshow_logpolars(fig, value)

                fname = "%s-%s.png" % (prefix, key)
                _savefig(fig, fname)
            # if "s-orig" in key:
            elif key == "amas-orig":
                center = np.array(reports["amas-result"], float)
                center[0] = 1.0 / center[0]
                reporting.imshow_pcorr(
                    fig, value, reports["amas-postproc"],
                    reports["amas-extent"], center,
                    reports["amas-success"], log_base=reports["base"]
                )
                fname = "%s-%s.png" % (prefix, key)
                _savefig(fig, fname)

        reporting.imshow_plain(fig, reports["asim"],
                               ("template", "sample", "tformed sample"))

        # Here goes a plot of template, rotated and scaled subject and
        fname = "{}-after-rot.png".format(prefix)
        _savefig(fig, fname)

        for idx in range(2):
            halves = np.array(ims[0].shape) / 2.0
            extent = np.array((- halves[1], halves[1], - halves[0], halves[0]))
            center = reports["t{}-tvec".format(idx)][::-1]
            img = reports["t{}-orig".format(idx)]
            reporting.imshow_pcorr(
                fig, img, reports["t{}-postproc".format(idx)],
                extent, center, reports["t{}-success".format(idx)]
            )

            fname = "{}-t{}.png".format(prefix, idx)
            _savefig(fig, fname)

        fig.clear()
        del fig

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
        resdict["unextended"] = _postprocess_unextend(ims, im2, opts["extend"])

    # We need this intact until now (for the transform above)
    resdict["tvec"] /= rcoef

    return resdict


def process_tile(ii, reports=None):
    global _SUCCS, _SHIFTS, _ANGLES, _SCALES, _DIFFS
    tile = _TILES[ii]
    image = _IMAGE
    opts = _OPTS
    pos = _POSS[ii]
    try:
        # TODO: Add unittests that zero success result
        #   doesn't influence anything
        with reporting.report_wrapper(reports, ii) as wrapped:
            resdict = process_images((tile, image), opts,
                                     reports=wrapped)
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
    if _SUCCS[ii] > 0:
        # print("%d: succ: %g" % (ii, resdict["success"]))
        # import pylab as pyl
        resdict["tvec"] -= pos
        tosa = ird.transform_img_dict(image, resdict, 0, opts["order"])
        tosa = utils.unextend_by(tosa, opts["extend"])
        # ird.imshow(tile, image, tosa, cmap=pyl.cm.gray)
        # pyl.show()
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


def settle_tiles(imgs, tiledim, opts, reports=None):
    global _SHIFTS
    coef = 0.35
    img0 = imgs[0]

    if reports is not None:
        slices = utils.getSlices(img0.shape, tiledim, coef)
        import pylab as pyl
        fig, axes = pyl.subplots()
        axes.imshow(img0)
        callback = reporting.Rect_mpl(axes)
        reporting.slices2rects(slices, callback)
        fig.savefig("tiling.png")

    tiles, poss = zip(* ird.utils.decompose(img0, tiledim, coef))

    _fill_globals(tiles, poss, imgs[1], opts)

    for ii, pos in enumerate(poss):
        process_tile(ii, reports)

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

    tosa_offset = np.array(img0.shape)[:2] - np.array(tiledim)[:2] + 0.5
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
                             opts["low"], opts["high"], opts["cut"],
                             opts["resample"])
    im2 = ird.transform_img_dict(ims[1], resdict, bgval, opts["order"])

    # TODO: This is kinda dirty
    resdict["unextended"] = _postprocess_unextend(ims, im2, opts["extend"])
    resdict["Dangle"], resdict["Dscale"], resdict["Dt"] = _DIFFS

    return resdict
