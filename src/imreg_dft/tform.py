# -*- coding: utf-8 -*-
# tform.py

# Copyright (c) 2014-?, Matěj Týč
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


import argparse as ap
import re

import numpy as np

import imreg_dft.cli as cli
import imreg_dft.loader as loader


def create_parser():
    parser = ap.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument("transformation", help="The transformation string.")
    cli.create_base_parser(parser)
    grp = parser.add_mutually_exclusive_group("Template shape")
    grp.add_argument("template", nargs="?")
    grp.add_argument("--template-shape")
    parser.add_argument("transformation", type=str2tform)
    loader.update_parser(parser)
    return parser


def str2tform(tstr):
    """
    Parses a transformation-descripting string to a transformation dict.
    """
    rexp = (
        "scale:\s*(?P<scale>\S*)\s*(+-\S*)?\s*"
        "angle:\s*(?P<angle>\S*)\s*(+-\S*)?\s*"
        "shift:\s*(?P<ty>[^,]*),\s*(?P<tx>[^,]*)\s*(+-\S*)?\s*"
        "success:\s*(?P<success>\S*)\s*"
    )
    match = re.search(rexp, tstr, re.MULTILINE)
    if match is None:
        raise ap.ArgumentTypeError()
    ret = dict()
    parsed = match.groupdict()
    for key, val in parsed.items():
        ret[key] = float(val)
    ret["tvec"] = np.array((ret["ty"], ret["tx"]))
    return ret


def args2dict(args):
    """
    Takes parsed command-line args and makes a dict that contains exact info
    about what needs to be done.
    """
    ret = dict()
    template_shape = None
    _loader = loader.LOADERS.get_loader(args.subject)
    ret["subject"] = _loader.load2reg(args.subject)
    if args.template is not None:
        img = _loader.load2reg(args.template)
        template_shape = img.shape
    elif args.template_shape is not None:
        template_shape = [int(x) for x in args.template_shape.split(",")]
    else:
        template_shape = ret["subject"].shape
    assert template_shape is not None, \
        "Template shape should have been determined by now, wtf that it wasn't"
    ret["shape"] = template_shape
    ret["tform"] = str2tform(args.transformation)
    return ret


def main():
    parser = create_parser()
    args = parser.parse_args()
    loader.settle_loaders(args)
