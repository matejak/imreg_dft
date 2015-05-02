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
import imreg_dft.utils as utils
import imreg_dft.loader as loader


def assure_constraint(possible_constraints):
    pass


def _constraints(what):
    BOUNDS = dict(
        angle=(-180, 180),
        scale=(0.5, 2.0),
    )

    def constraint(string):
        components = string.split(",")
        if not (0 < len(components) <= 2):
            raise ap.ArgumentTypeError(
                "We accept at most %d (but at least 1) comma-delimited numbers,"
                " you have passed us %d" % (len(components), 2))
        try:
            mean = float(components[0])
        except Exception:
            raise ap.ArgumentTypeError(
                "The %s value must be a float number, got '%s'."
                % (what, components[0]))
        if what in BOUNDS:
            lo, hi = BOUNDS[what]
            if not lo <= mean <= hi:
                raise ap.ArgumentTypeError(
                    "The %s value must be a number between %g and %g, got %g."
                    % (lo, hi, mean))
        if len(components) == 2:
            std = components[1]
            if len(std) == 0:
                std = None
            else:
                try:
                    std = float(std)
                except Exception:
                    raise ap.ArgumentTypeError(
                        "The %s standard deviation spec must be either"
                        "either a float number or nothing, got '%s'."
                        % (what, std))
        ret = (mean, std)
        return ret
    return constraint


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
    fake_data = dict(scale=1.0, angle=2.0, tx=2, ty=2,
                     Dscale=0.1, Dangle=0.2, Dt=0.5, success=0.99)
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


def create_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('template')
    parser.add_argument('subject')
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
    # parser.add_argument('--exponent', type=_exponent, default="inf",
    #                     help="Either 'inf' or float. See the docs.")
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
                        help="Where to save the transformed subject.")
    parser.add_argument(
        '--filter-pcorr', type=int, default=0,
        help="Whether to filter during translation detection. Normally not "
        "needed, but when using low-pass filtering, you may need to increase "
        "filter radius (0 means no filtering, 4 should be enough)")
    parser.add_argument(
        '--print-result', action="store_true", default=False,
        help="We don't print anything unless this option is specified")
    parser.add_argument(
        '--print-format', default="scale: %(scale)f +-%(Dscale)g\n"
        "angle: %(angle)f +-%(Dangle)g\n"
        "shift: %(tx)g, %(ty)g +-%(Dt)g\nSuccess: %(success).3g\n", type=outmsg,
        help="Print a string (to stdout) in a given format. A dictionary "
        "containing the 'scale', 'angle', 'tx', 'ty', 'Dscale', 'Dangle', "
        "'Dt' and 'success' keys will be passed for string interpolation")
    parser.add_argument(
        '--tile', action="store_true", default=False, help="If the template "
        "is larger than the subject, break the template to pieces of size "
        "similar to subject size.")
    parser.add_argument('--version', action="version",
                        version="imreg_dft %s" % ird.__version__,
                        help="Just print version and exit")
    parser.add_argument(
        "--angle", type=_constraints("angle"),
        metavar="MEAN[,STD]", default=(0, None),
        help="The mean and standard deviation of the expected angle. ")
    parser.add_argument(
        "--scale", type=_constraints("scale"),
        metavar="MEAN[,STD]", default=(1, None),
        help="The mean and standard deviation of the expected scale. ")
    parser.add_argument(
        "--tx", type=_constraints("shift"),
        metavar="MEAN[,STD]", default=(0, None),
        help="The mean and standard deviation of the expected X translation. ")
    parser.add_argument(
        "--ty", type=_constraints("shift"),
        metavar="MEAN[,STD]", default=(0, None),
        help="The mean and standard deviation of the expected Y translation. ")
    loader.update_parser(parser)
    return parser


def main():
    parser = create_parser()

    args = parser.parse_args()

    loader_stuff = loader.settle_loaders(args, (args.template, args.subject))

    # We need tuples in the parser and lists further in the code.
    # So we have to do it like this.
    constraints = dict(
        angle=list(args.angle),
        scale=list(args.scale),
        tx=list(args.tx),
        ty=list(args.ty),
    )

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
        exponent="inf",
        resample=args.resample,
        tile=args.tile,
        constraints=constraints,
        output=args.output,
    )
    opts.update(loader_stuff)
    run(args.template, args.subject, opts)


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


def _get_resdict(imgs, opts, tosa=None):
    import numpy as np

    tiledim = None
    if opts["tile"]:
        shapes = np.array([np.array(img.shape) for img in imgs])
        if (shapes[0] / shapes[1]).max() > 1.7:
            tiledim = np.min(shapes, axis=0) * 1.1
            # TODO: Establish a translate region constraint of width tiledim * coef

    if tiledim is not None:
        tiles = ird.utils.decompose(imgs[0], tiledim, 0.35)
        resdicts = []
        shifts = np.zeros((len(tiles), 2), float)
        succs = np.zeros(len(tiles), float)
        angles = np.zeros(len(tiles), float)
        scales = np.zeros(len(tiles), float)
        for ii, (tile, pos) in enumerate(tiles):
            try:
                # TODO: Add unittests that zero success result
                #   doesn't influence anything
                resdict = process_images((tile, imgs[1]), opts, None)
                angles[ii] = resdict["angle"]
                scales[ii] = resdict["scale"]
                shifts[ii] = np.array((resdict["ty"], resdict["tx"])) + pos
            except ValueError:
                # probably incompatible images due to high scale change, so we
                # just add some harmless stuff here and proceed.
                resdict = dict(success=0)
            resdicts.append(resdict)
            succs[ii] = resdict["success"]
            if 0:
                print(ii, succs[ii])
                import pylab as pyl
                pyl.figure(); pyl.imshow(tile)
                pyl.show()
        tosa_offset = np.array(imgs[0].shape)[:2] - np.array(tiledim)[:2] + 0.5
        shifts -= tosa_offset / 2.0

        # Get the cluster of the tiles that have similar results and that look
        # most promising along with the index of the best tile
        cluster, amax = utils.get_best_cluster(shifts, succs, 5)
        # Make the quantities estimation even more precise by taking
        # the average of all good tiles
        shift, angle, scale, score = utils.get_values(
            cluster, shifts, succs, angles, scales)

        resdict = resdicts[amax]

        resdict["scale"] = scale
        resdict["angle"] = angle
        resdict["tvec"] = shift
        resdict["ty"], resdict["tx"] = resdict["tvec"]

        # In non-tile cases, tosa is transformed in process_images
        if tosa is not None:
            tosa = ird.transform_img_dict(tosa, resdict)
    else:
        resdict = process_images(imgs, opts, tosa)

    return resdict


def run(template, subject, opts):
    # lazy import so no imports before run() is really called
    from imreg_dft import imreg

    fnames = (template, subject)
    loaders = opts["loaders"]
    loader_img = loaders[1]
    imgs = [loa.load2reg(fname) for fname, loa in zip(fnames, loaders)]

    tosa = None
    saver = None
    outname = opts["output"]
    if outname is not None:
        tosa = loader_img.get2save()
        saver = loader.LOADERS.get_loader(outname)
        tosa = ird.utils.extend_to_3D(tosa, imgs[0].shape[:3])

    resdict = _get_resdict(imgs, opts, tosa)
    im0, im1, im2 = resdict['unextended']

    if opts["print_format"] is not None:
        msg = opts["print_format"] % resdict
        msg = msg.encode("utf-8")
        msg = msg.decode('unicode_escape')
        sys.stdout.write(msg)

    if outname is not None:
        saver.save(outname, tosa, loader_img)

    if opts["show"]:
        # import ipdb; ipdb.set_trace()
        import pylab as pyl
        fig = pyl.figure()
        imreg.imshow(im0, im1, im2, fig=fig)
        # imreg.imshow(imgs[0], imgs[1], im2, fig=fig)
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
        ims[0], ims[1], opts["iters"], opts["order"], opts["constraints"],
        opts["filter_pcorr"], opts["exponent"])

    im2 = resdict.pop("timg")

    # Seems that the reampling simply scales the translation
    resdict["tvec"] /= rcoef
    ty, tx = resdict["tvec"]
    resdict["tx"] = tx
    resdict["ty"] = ty
    resdict["imgs"] = ims
    tform = resdict

    if tosa is not None:
        tosa[:] = ird.transform_img_dict(tosa, tform)

    if rcoef != 1:
        ims = [resample(img, 1.0 / rcoef) for img in ims]
        im2 = resample(im2, 1.0 / rcoef)
        resdict["Dt"] /= rcoef

    resdict["unextended"] = [utils.unextend_by(img, opts["extend"])
                             for img in ims + [im2]]

    return resdict


if __name__ == "__main__":
    main()
