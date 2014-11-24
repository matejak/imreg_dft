import os

import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt

import imreg_dft as ird


basedir = os.path.join('..', 'examples')
# the TEMPLATE
im0 = sp.misc.imread(os.path.join(basedir, "sample1.png"), True)
# the image to be transformed
im1 = sp.misc.imread(os.path.join(basedir, "sample3.png"), True)
result = ird.similarity(im0, im1, numiter=3)
ird.imshow(im0, im1, result['timg'])
plt.show()
