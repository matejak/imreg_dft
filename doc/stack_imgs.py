import argparse as ap

import scipy as sp
import scipy.misc

import matplotlib

matplotlib.use('Cairo')

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


_LABELS = "abcde"


def parse():
    parser = ap.ArgumentParser()
    parser.add_argument(
        "infiles", nargs="+")
    parser.add_argument(
        "-s", "--size", default=[5.8, 2.8],
        type=lambda x: [float(y) for y in x.split(",")],
        help="Size of the image (inches)")
    parser.add_argument(
        "--colormap", default="gray",
        help="Name of the colormap (in matplotlib.cm namespace)")
    parser.add_argument(
        "-d", "--dpi", default=200.0, type=float,
        help="Resolution of the image")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Where to save the result")
    ret = parser.parse_args()
    return ret


def _imshow(pl, what, label, cmap):
    pl.grid()
    pl.imshow(what, cmap=cmap)
    pl.tick_params(axis='both', which='major', labelsize=10)
    at = AnchoredText(
        label, loc=2)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    pl.add_artist(at)


def mkFig(fig, infiles, cmap):
    imgs = [sp.misc.imread(fname, True) for fname in infiles]
    ncols = len(imgs)
    pl0 = fig.add_subplot(1, ncols, 1)
    _imshow(pl0, imgs[0], _LABELS[0], cmap)
    for ii, img in enumerate(imgs[1:]):
        ii += 2
        pl = fig.add_subplot(1, ncols, ii, sharey=pl0)
        plt.setp(pl.get_yticklabels(), visible=False)
        _imshow(pl, img, _LABELS[ii - 1], cmap)


def main():
    args = parse()
    cmap = getattr(matplotlib.cm, args.colormap)
    fig = plt.Figure(dpi=args.dpi, figsize=args.size)
    mkFig(fig, args.infiles, cmap)
    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
