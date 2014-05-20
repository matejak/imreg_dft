# -*- coding: utf-8 -*-
import argparse as ap

import numpy as np
from scipy import misc
import pylab as pyl

from imreg_dft import utils
from imreg_dft import imreg


def _float_tuple(st):
    vals = st.split(",")
    if len(vals) != 2:
        raise Exception("'%s' are not two values delimited by comma" % st)
    try:
        vals = [float(val) for val in vals]
    except ValueError:
        raise Exception("%s are not two float values" % vals)
    return vals


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('image')
    parser.add_argument('--show', action="store_true", default=False,
                        help="Whether to show registration result")
    parser.add_argument('--lowpass', type=_float_tuple,
                        action="append", default=[])
    parser.add_argument('--highpass', type=_float_tuple,
                        action="append", default=[])
    parser.add_argument('--apodize', type=float,
                        default=0.2)
    args = parser.parse_args()

    opts = dict(
        aporad=args.apodize,
        low=args.lowpass,
        high=args.highpass,
        show=args.show,
    )
    run(args.template, args.image,
        opts)


def filter_images(ims, low, high):
    ret = [utils.filter(im, low, high) for im in ims]
    return ret


def apodize(ims, radius_ratio):
    ret = []
    # They might have different shapes...
    for im in ims:
        shape = im.shape

        bgval = np.median(im)
        bg = np.ones(shape) * bgval

        radius = radius_ratio * min(shape)
        apofield = utils.get_apofield(shape, radius)

        bg *= (1 - apofield)
        # not modifying inplace
        toapp = bg + im * apofield
        ret.append(toapp)
    return ret


def run(template, image, opts):

    ims = [misc.imread(fname, True) for fname in (template, image)]
    bigshape = np.array([im.shape for im in ims]).max(0)
    ims = apodize(ims, opts["aporad"])
    ims = filter_images(ims, opts["low"], opts["high"])
    # We think that im[0, 0] has the right value after apodization
    ims = [imreg.embed_to(np.zeros(bigshape) + im[0, 0], im) for im in ims]

    im2, scale, angle, (t0, t1) = imreg.similarity(* ims)
    print("scale: %f" % scale)
    print("angle: %f" % angle)
    print("shift: %d,%d" % (t0, t1))

    if opts["show"]:
        imreg.imshow(ims[0], ims[1], im2)
        pyl.show()


if __name__ == "__main__":
    main()
