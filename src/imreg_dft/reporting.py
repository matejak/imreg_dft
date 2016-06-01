# -*- coding: utf-8 -*-

# reporting.py

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

import contextlib

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axg


@contextlib.contextmanager
def report_wrapper(orig, index):
    if orig is None:
        yield None
    else:
        ret = ReportsWrapper(orig)
        ret.push_index(index)
        yield ret
        ret.pop_index(index)


class ReportsWrapper(object):
    """
    A wrapped dictionary.
    It allows a parent function to put it in a mode, in which it will
    prefix keys of items set.
    """
    def __init__(self, reports):
        assert reports is not None, \
            ("Use the report_wrapper wrapper factory, don't "
             "create wrappers from None")
        self.reports = reports
        self.prefixes = []
        self.idx = ""

    def keys(self):
        return self.reports.keys()

    def pop(self, what):
        return self.reports.pop(what)

    def items(self):
        return self.reports.items()

    def __setitem__(self, key, value):
        key = self.idx + key
        self.reports[key] = value

    def __getitem__(self, key):
        assert self.reports is not None
        return self.reports[key]

    def _idx2prefix(self, idx):
        ret = "%03d-" % idx
        return ret

    def push_index(self, idx):
        prefix = self._idx2prefix(idx)
        self.push_prefix(prefix)

    def pop_index(self, idx):
        prefix = self._idx2prefix(idx)
        self.pop_prefix(prefix)

    def push_prefix(self, idx):
        self.prefixes.append(idx)
        self.idx = "%s" % idx

    def pop_prefix(self, idx):
        assert self.prefixes[-1] == idx
        self.prefixes.pop()
        if len(self.prefixes) > 0:
            self.idx = self.prefixes[-1]
        else:
            self.idx = ""


class Rect_callback(object):
    def __call__(self, idx, LLC, dims):
        self._call(idx, LLC, dims)

    def _call(idx, LLC, dims):
        raise NotImplementedError()


class Rect_mpl(Rect_callback):
    """
    A class that can draw image tiles nicely
    """
    def __init__(self, subplot):
        self.subplot = subplot

    def _call(self, idx, LLC, dims, special=False):
        # We don't want to import this on the top-level due to test stuff
        import matplotlib.pyplot as plt
        # Get from the numpy -> MPL coord system
        LLC = LLC[::-1]
        URC = LLC + np.array((dims[1], dims[0]))
        kwargs = dict(fc='none')
        if special:
            kwargs["fc"] = 'w'
            kwargs["alpha"] = 0.5
        rect = plt.Rectangle(LLC, dims[1], dims[0], ** kwargs)
        self.subplot.add_artist(rect)
        center = (URC + LLC) / 2.0
        self.subplot.text(center[0], center[1], "(%02d)" % idx)


def slices2rects(slices, rect_cb):
    """
    Args:
        slices: List of slice objects
        rect_cb (callable): Check :class:`Rect_callback`.
    """
    for ii, (sly, slx) in enumerate(slices):
        LLC = np.array((sly.start, slx.start))
        URC = np.array((sly.stop,  slx.stop))
        dims = URC - LLC
        rect_cb(ii, LLC, dims)


def imshow_spectra(fig, spectra):
    dfts_filt_extent = (-0.5, 0.5, -0.5, 0.5)
    grid = axg.ImageGrid(
        fig, 111, nrows_ncols=(1, 2),
        add_all=True,
        axes_pad=0.4, label_mode="L",
        cbar_pad=0.05, cbar_mode="each", cbar_size="3.5%",
    )
    what = ("template", "subject")
    for ii, im in enumerate(spectra):
        grid[ii].set_title("log abs dfts - %s" % what[ii])
        im = grid[ii].imshow(np.log(np.abs(im)), cmap=plt.cm.viridis,
                             extent=dfts_filt_extent, )
        grid.cbar_axes[ii].colorbar(im)
    return fig


def imshow_logpolars(fig, spectra):
    logpolars_extent = (0, 0.5, 0, 180)
    grid = axg.ImageGrid(
        fig, 111, nrows_ncols=(2, 1),
        add_all=True,
        aspect=False,
        axes_pad=0.4, label_mode="L",
        cbar_pad=0.05, cbar_mode="each", cbar_size="3.5%",
    )
    ims = [np.log(np.abs(im)) for im in spectra]
    vmin = min([np.percentile(im, 2) for im in ims])
    vmax = max([np.percentile(im, 98) for im in ims])
    for ii, im in enumerate(ims):
        im = grid[ii].imshow(im, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax,
                             aspect="auto", extent=logpolars_extent)
        grid.cbar_axes[ii].colorbar(im)

    return fig


def imshow_plain(fig, images, what, also_common=False):
    ncols = len(images)
    nrows = 1
    if also_common:
        nrows = 2
    grid = axg.ImageGrid(
        fig, 111,  nrows_ncols=(nrows, ncols), add_all=True,
        axes_pad=0.4, label_mode="L",
        cbar_pad=0.05, cbar_mode="each", cbar_size="3.5%",
    )
    images = [im.real for im in images]

    for ii, im in enumerate(images):
        vmin = np.percentile(im, 2)
        vmax = np.percentile(im, 98)
        grid[ii].set_title("individual cmap --- {}".format(what[ii]))
        img = grid[ii].imshow(im, cmap=plt.cm.gray, origin="lower",
                              vmin=vmin, vmax=vmax)
        grid.cbar_axes[ii].colorbar(img)

    if also_common:
        vmin = min([np.percentile(im, 2) for im in images])
        vmax = max([np.percentile(im, 98) for im in images])
        for ii, im in enumerate(images):
            grid[ii + ncols].set_title("common cmap --- {}".format(what[ii]))
            im = grid[ii + ncols].imshow(im, cmap=plt.cm.viridis,
                                         origin="lower", vmin=vmin, vmax=vmax)
            grid.cbar_axes[ii + ncols].colorbar(im)

    return fig


def imshow_pcorr(fig, raw, filtered, extent, result, success, log_base=None):
    grid = axg.ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(1, 2),
        add_all=True,
        axes_pad=0.4,
        aspect=False,
        cbar_pad=0.05,
        label_mode="L",
        cbar_mode="single",
        cbar_size="3.5%",
    )
    vmax = raw.max()
    imshow_kwargs = dict(
        vmin=0, vmax=vmax,
        aspect="auto",
        origin="lower", extent=extent,
        cmap=plt.cm.viridis,
    )
    grid[0].set_title("pcorr --- original")
    labels = ("translation y", "translation x")
    if log_base is not None:
        for dim in range(2):
            grid[dim].set_xscale("log", basex=log_base)
            grid[dim].get_xaxis().set_major_formatter(plt.ScalarFormatter())
        labels = ("rotation / degrees", "scale change")
    grid[0].set_ylabel(labels[0])
    grid[0].imshow(raw, ** imshow_kwargs)

    center = np.array(result)
    # Otherwise plot would change xlim
    grid[0].autoscale(False)
    grid[0].plot(center[0], center[1], "o",
                 color="r", fillstyle="none", markersize=18, lw=8)
    grid[0].annotate("succ: {:.3g}".format(success), xy=center,
                     xytext=(0, 8), textcoords='offset points',
                     color="red", va="bottom", ha="center")
    grid[1].set_title("pcorr --- constrained and filtered")
    im = grid[1].imshow(filtered, ** imshow_kwargs)
    grid.cbar_axes[0].colorbar(im)

    # The common stuff
    for idx in range(2):
        grid[idx].grid(c="w")
        grid[idx].set_xlabel(labels[1])

    """
    pl = fig.add_subplot(122)
    extent2 = np.zeros(4, int)
    radius = 8
    center = (np.array(reports["amas-result-raw"], int)
                + np.array(raw.shape, int) // 2)
    closeup = utils._get_subarr(raw, center, radius)

    pl.imshow(closeup, interpolation="nearest", origin="lower",
              cmap=plt.cm.viridis, vmin=0, vmax=vmax)
    """

    return fig
