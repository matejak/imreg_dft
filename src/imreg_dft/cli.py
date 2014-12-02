# -*- coding: utf-8 -*-
# cli.py

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
FFT based image registration. --- CLI frontend
"""

import sys
import argparse as ap

import imreg_dft as ird
import imreg_dft.loader


def _float_tuple(string):
    """
    Support function for parsing string of two floats delimited by a comma.
    """
    vals = string.split(",")
    if len(vals) != 2:
        raise ap.ArgumentTypeError(
            "'%s' are not two values delimited by comma" % string)
    try:
        vals = [float(val) for val in vals]
    except ValueError:
        raise ap.ArgumentTypeError("%s are not two float values" % vals)
    return vals


def _exponent(string):
    """
    Converts the passed string to a float or "inf"
    """
    if string == 'inf':
        return string
    try:
        ret = float(string)
    except:
        raise ap.ArgumentTypeError(
            "'%s' should be either 'inf' or a float value" % string)
    return ret


def outmsg(msg):
    """
    Support function for checking of validity of the output format string.
    A test interpolation is performed and exceptions handled.
    """
    fake_data = dict(scale=1.0, angle=2.0, tx=2, ty=2)
    tpl = "The string '%s' is not a good format string"
    try:
        msg % fake_data
    except KeyError as exc:
        raise ap.ArgumentTypeError(
            (tpl + ". The correct string "
             "has to contain at most %s, but this one also contains an invalid"
             " value '%s'.") % (msg, fake_data.keys(), exc.args[0]))
    except Exception as exc:
        raise ap.ArgumentTypeError(
            (tpl + " - %s") % (msg, exc.message))
    return msg


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('image')
    parser.add_argument('--show', action="store_true", default=False,
                        help="Whether to show registration result")
    parser.add_argument('--lowpass', type=_float_tuple,
                        default=None, metavar="HI_THRESH,LOW_THRESH",
                        help="1,1 means no-op, 0.8,0.9 is a mild filter")
    parser.add_argument('--highpass', type=_float_tuple,
                        default=None, metavar="HI_THRESH,LOW_THRESH",
                        help="0,0 means no-op, 0.1,0.2 is a mild filter")
    parser.add_argument('--resample', type=float, default=1,
                        help="Work with resampled images.")
    parser.add_argument('--exponent', type=_exponent, default="inf",
                        help="Either 'inf' or float. See the docs.")
    parser.add_argument(
        '--invert', action="store_true", default=False,
        help="Whether to invert the template. Don't expect much from it tho.")
    parser.add_argument('--iters', type=int, default=1,
                        help="How many iterations to guess the right scale "
                        "and angle")
    parser.add_argument('--extend', type=int, metavar="PIXELS", default=0,
                        help="Extend images by the specified amount of pixels "
                        "before the processing (thus eliminating "
                        "edge effects)")
    parser.add_argument('--order', type=int, default=1,
                        help="Interpolation order (1 = linear, 3 = cubic etc)")
    parser.add_argument('--output', '-o',
                        help="Where to save the transformed image.")
    parser.add_argument(
        '--filter-pcorr', type=int, default=0,
        help="Whether to filter during translation detection. Normally not "
        "needed, but when using low-pass filtering, you may need to increase "
        "filter radius (0 means no filtering, 4 should be enough)")
    parser.add_argument(
        '--print-result', action="store_true", default=False,
        help="We don't print anything unless this option is specified")
    parser.add_argument(
        '--print-format', default="scale: %(scale)f\nangle: %(angle)f\nshift: "
        "%(tx)g, %(ty)g\n", type=outmsg,
        help="Print a string (to stdout) in a given format. A dictionary "
        "containing the 'scale', 'angle', 'tx' and 'ty' keys will be "
        "passed for interpolation")
    parser.add_argument('--version', action="version",
                        version="imreg_dft %s" % ird.__version__,
                        help="Just print version and exit")
    ird.loader.update_parser(parser)

    args = parser.parse_args()

    loader_stuff = ird.loader.settle_loaders(args, (args.template, args.image))

    print_format = args.print_format
    if not args.print_result:
        print_format = None

    opts = dict(
        order=args.order,
        filter_pcorr=args.filter_pcorr,
        extend=args.extend,
        low=args.lowpass,
        high=args.highpass,
        show=args.show,
        print_format=print_format,
        iters=args.iters,
        exponent=args.exponent,
        resample=args.resample,
        invert=args.invert,
        output=args.output,
    )
    opts.update(loader_stuff)
    run(args.template, args.image, opts)


def filter_images(imgs, low, high):
    # lazy import so no imports before run() is really called
    from imreg_dft import utils

    ret = [utils.imfilter(img, low, high) for img in imgs]
    return ret


def apodize(imgs, radius_ratio):
    # lazy import so no imports before run() is really called
    import numpy as np
    from imreg_dft import utils

    ret = []
    # They might have different shapes...
    for img in imgs:
        shape = img.shape

        bgval = np.median(img)
        bg = np.ones(shape) * bgval

        radius = radius_ratio * min(shape)
        apofield = utils.get_apofield(shape, radius)

        bg *= (1 - apofield)
        # not modifying inplace
        toapp = bg + img * apofield
        ret.append(toapp)
    return ret


def run(template, image, opts):
    # lazy import so no imports before run() is really called
    from imreg_dft import imreg

    loaders = ird.loader.LOADERS

    fnames = (template, image)
    loaders = opts["loaders"]
    imgs = [loa.load2reg(fname) for fname, loa in zip(fnames, loaders)]

    tosa = None
    if opts["output"] is not None:
        tosa = loaders[1].get2save()

    if opts["invert"]:
        imgs[0] *= -1
    im2 = process_images(imgs, opts, tosa)

    if opts["output"] is not None:
        loaders[1].save(opts["output"], tosa)

    if opts["show"]:
        import pylab as pyl
        fig = pyl.figure()
        imreg.imshow(imgs[0], imgs[1], im2, fig=fig)
        pyl.show()


def resample(img, coef):
    from scipy import signal
    ret = img
    for axis in range(2):
        newdim = ret.shape[axis] * coef
        ret = signal.resample(ret, newdim, axis=axis)
    return ret


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
        ims[0], ims[1], opts["iters"], opts["order"],
        opts["filter_pcorr"], opts["exponent"])

    im2 = resdict.pop("timg")

    # Seems that the reampling simply scales the translation
    resdict["tvec"] /= rcoef
    ty, tx = resdict["tvec"]
    resdict["tx"] = tx
    resdict["ty"] = ty
    tform = resdict

    if tosa is not None:
        tosa[:] = ird.transform_img_dict(tosa, tform)

    if opts["print_format"] is not None:
        msg = opts["print_format"] % tform
        msg = msg.encode("utf-8")
        msg = msg.decode('unicode_escape')
        sys.stdout.write(msg)

    if rcoef != 1:
        im2 = resample(im2, 1 / rcoef)

    im2 = utils.unextend_by(im2, opts["extend"])

    return im2


if __name__ == "__main__":
    main()
