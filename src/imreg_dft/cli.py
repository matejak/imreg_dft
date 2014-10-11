# -*- coding: utf-8 -*-
import argparse as ap


def _float_tuple(st):
    vals = st.split(",")
    if len(vals) != 2:
        raise ap.ArgumentTypeError("'%s' are not two values delimited by comma"
                                   % st)
    try:
        vals = [float(val) for val in vals]
    except ValueError:
        raise ap.ArgumentTypeError("%s are not two float values" % vals)
    return vals


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('image')
    parser.add_argument('--show', action="store_true", default=False,
                        help="Whether to show registration result")
    parser.add_argument('--lowpass', type=_float_tuple, action="append",
                        default=[], metavar="HI_THRESH,LOW_THRESH",
                        help="1,1 means no-op, 0.9,0.8 is a mild filter")
    parser.add_argument('--highpass', type=_float_tuple, action="append",
                        default=[], metavar="HI_THRESH,LOW_THRESH",
                        help="0,0 means no-op, 0.2,0.1 is a mild filter")
    parser.add_argument('--extend', type=int, metavar="PIXELS", default=0,
                        help="Extend images by the specified amount of pixels "
                        "before the processing (thus eliminating edge effects)")
    parser.add_argument('--order', type=int, default=1,
                        help="Interpolation order (1 = linear, 3 = cubic etc.)")
    parser.add_argument(
        '--filter-pcorr', type=int, default=0,
        help="Whether to filter during translation detection. Normally not "
        "needed, but when using low-pass filtering, you may need to increase "
        "filter radius (0 means no filtering, 4 should be enough)")
    args = parser.parse_args()

    opts = dict(
        order=args.order,
        filter_pcorr=args.filter_pcorr,
        extend=args.extend,
        low=args.lowpass,
        high=args.highpass,
        show=args.show,
    )
    run(args.template, args.image, opts)


def filter_images(ims, low, high):
    from imreg_dft import utils

    ret = [utils.filter(im, low, high) for im in ims]
    return ret


def apodize(ims, radius_ratio):
    import numpy as np
    from imreg_dft import utils

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
    import numpy as np
    from scipy import misc

    from imreg_dft import utils
    from imreg_dft import imreg

    ims = [misc.imread(fname, True) for fname in (template, image)]
    bigshape = np.array([im.shape for im in ims]).max(0)
    ims = [utils.extend_by(im, opts["extend"]) for im in ims]
    ims = filter_images(ims, opts["low"], opts["high"])
    # We think that im[0, 0] has the right value after apodization
    ims = [utils.embed_to(np.zeros(bigshape) + im[0, 0], im) for im in ims]

    im2, scale, angle, (t0, t1) = imreg.similarity(
        ims[0], ims[1], opts["order"], opts["filter_pcorr"])
    print("scale: %f" % scale)
    print("angle: %f" % angle)
    print("shift: %d,%d" % (t0, t1))

    ims = [utils.unextend_by(im, opts["extend"]) for im in ims]
    im2 = utils.unextend_by(im2, opts["extend"])

    if opts["show"]:
        import pylab as pyl
        imreg.imshow(ims[0], ims[1], im2)
        pyl.show()


if __name__ == "__main__":
    main()
