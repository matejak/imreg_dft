# -*- coding: utf-8 -*-
import argparse as ap

from scipy import misc
import pylab as pyl

import imreg


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('image')
    parser.add_argument('--show', action="store_true", default=False,
                        help="Whether to show registration result")
    args = parser.parse_args()

    run(args.template, args.image,
        dict(show=args.show))


def run(template, image, opts):

    ims = [misc.imread(fname, True) for fname in (template, image)]
    im2, scale, angle, (t0, t1) = imreg.similarity(* ims)
    print("scale: %f" % scale)
    print("angle: %f" % angle)
    print("shift: %d,%d" % (t0, t1))

    if opts["show"]:
        imreg.imshow(ims[0], ims[1], im2)
        pyl.show()


if __name__ == "__main__":
    main()
