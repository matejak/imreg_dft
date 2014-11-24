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
t0, t1 = ird.translation(im0, im1)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=(t0, t1))
ird.imshow(im0, im1, timg)
plt.show()
print(t0, t1)
