import unittest as ut

import numpy as np
import numpy.fft
import numpy.testing

import imreg_dft.imreg as imreg


class TestImreg(ut.TestCase):
    def testCalcCog(self):
        src = np.array([[3, 1, 3.01],
                        [1, 1, 1],
                        [0, 0, 0]])
        src = np.fft.ifftshift(src)
        # After FFTShift:
        # [[ 1.    1.    1.  ]
        #  [ 0.    0.    0.  ]
        #  [ 1.    3.01  3.  ]]
        infres = imreg._calc_cog(src, 'inf')  # element 3.01
        self.assertEqual(tuple(infres), (2, 1))
        n10res = imreg._calc_cog(src, 10)  # element 1 in the rows with 3s
        self.assertEqual(tuple(n10res), (2, 0))


if __name__ == '__main__':
    ut.main()
