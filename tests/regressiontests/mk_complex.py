import numpy as np

import scipy as sp
import scipy.io
import scipy.signal


np.random.seed(4)
abs_val, phase_val = [sp.rand(13, 20) for _ in range(2)]
phase_val *= 2 * np.pi
shift = (2, 3)

for img in (abs_val, phase_val):
    for ax in range(2):
        img[:] = sp.signal.resample(img, int(img.shape[ax] * 1.5), axis=ax)

cplx = dest * np.exp(1j * np.pi * 2 * dest2)

first = cplx[shift[0]:, shift[1]:]
second = cplx[:-shift[0], :-shift[1]]
sp.io.savemat("first.mat", dict(rca=first))
sp.io.savemat("first2.mat", dict(rca=second))
