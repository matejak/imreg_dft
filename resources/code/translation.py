import os

import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt

import imreg_dft as ird


basedir = os.path.join('..', 'examples')
# the TEMPLATE
im0 = sp.misc.imread(os.path.join(basedir, "sample1.png"), True)
# the image to be transformed
im1 = sp.misc.imread(os.path.join(basedir, "sample2.png"), True)
tvec, succ, _ = ird.translation(im0, im1)
tvec = tvec.round(4)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)

# Maybe we don't want to show plots all the time
if not os.environ.get("IMSHOW", "yes") == "no":
    ird.imshow(im0, im1, timg)
    plt.show()

print("Translation is {}, success rate {:.4g}".format(tuple(tvec), succ))
