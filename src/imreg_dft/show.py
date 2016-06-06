# -*- coding: utf-8 -*-
# show.py

# Copyright (c) 2016-?, Matěj Týč
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

from imreg_dft import cli
from imreg_dft import reporting


TOSHOW = (
    "filtered input (I)mages",
    "filtered input images (S)pectra",
    "spectra (L)ogpolar transform",
    "(1) angle-scale phase correlation",
    "angle-scale transform (A)pplied",
    "(2) translation phase correlation",
    "(T)ile info",
)


TOSHOW_ABBR = "isl1a2t"


def create_parser():
    parser = ap.ArgumentParser()
    cli.update_parser_imreg(parser)
    parser.add_argument("--prefix", default="reports")
    parser.add_argument("--ftype", choices=("png", "pdf"), default="png")
    parser.add_argument(
        "--display", type=_show_valid, default=TOSHOW_ABBR,
        help="String composing of '{}', meaning respectively: {}."
        .format(TOSHOW_ABBR, ", ".join(TOSHOW)))
    return parser


def _show_valid(stri):
    stripped = stri.rstrip(TOSHOW_ABBR)
    if len(stripped) > 0:
        raise ap.ArgumentError("Argument contains invalid characters: {}"
                               .format(stripped))
    return stri


def main():
    parser = create_parser()

    args = parser.parse_args()

    opts = cli.args2dict(args)
    reports = reporting.ReportsWrapper(dict(), args.display)
    opts["show"] = False
    opts["reports"] = reports
    opts["prefix"] = args.prefix
    cli.run(args.template, args.subject, opts)


if __name__ == "__main__":
    main()
