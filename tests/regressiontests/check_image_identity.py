import sys
import argparse as ap

import numpy as np
from scipy import misc


STR2NORM = dict(
    max=np.max,
    mean=np.mean,
)


RET2RES = ("same", "different")


def main():
    parser = ap.ArgumentParser()
    parser.add_argument("one")
    parser.add_argument("two")
    parser.add_argument("thresh", type=float, default=0.01, nargs="?")
    parser.add_argument("--grayscale", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--norm", choices=("max", "mean"), default="max")
    args = parser.parse_args()

    imgs = [misc.imread(fname, args.grayscale).astype(float)
            for fname in (args.one, args.two)]
    diff = np.abs(imgs[1] - imgs[0])
    mean = np.abs(imgs[0] + imgs[1]) / 2.0
    if not args.grayscale:
        # format (y, x, RGB)
        diff, mean = [np.sum(arr, axis=-1) / 3.0 for arr in (diff, mean)]

    norm = STR2NORM[args.norm]
    if args.verbose:
        print("Reference norm: {:.4g}\nDifference norm: {:.4g}"
              .format(norm(mean), norm(diff)))
    ret = 0
    if norm(mean) * args.thresh < norm(diff):
        # The difference is too big
        ret = 1

    if args.verbose:
        print("The two images are {}.".format(RET2RES[ret]))

    sys.exit(ret)


if __name__ == "__main__":
    main()
