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


@contextlib.contextmanager
def report_wrapper(orig, index):
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
        self.reports = reports
        self.prefixes = []
        self.idx = ""

    def pop(self, what):
        return self.reports.pop(what)

    def items(self):
        if self.reports is None:
            return []
        else:
            return self.reports.items()

    def __setitem__(self, key, value):
        if self.reports is None:
            return
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
