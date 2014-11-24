import argparse as ap

import scipy as sp
import scipy.misc
from matplotlib import pyplot as plt


def parse():
    parser = ap.ArgumentParser()
    parser.add_argument(
        "infiles", nargs="+")
    parser.add_argument(
        "-s", "--size", default=[5, 4],
        type=lambda x: [float(y) for y in x.split(",")],
        help="Size of the image (inches)")
    parser.add_argument(
        "-d", "--dpi", default=150.0, type=float,
        help="Resolution of the image")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Where to save the result")
    ret = parser.parse_args()
    return ret


def mkFig(fig, infiles):
    imgs = [sp.misc.imread(fname) for fname in infiles]
    for ii, img in enumerate(imgs):
        pl = fig.add_subplot(ii, 1, ii)
        pl.imshow(img)


def run():
    args = parse()
    fig = plt.Figure(dpi=args.dpi, size=args.size)
    fig.savefig(args.output)


    if __name__ == "__main__":
        run()
