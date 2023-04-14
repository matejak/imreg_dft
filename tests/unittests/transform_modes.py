from __future__ import print_function, division
import unittest as ut

import imreg_dft.imreg as imreg
import imreg_dft.utils as utils
import numpy.testing as npt

from scipy.misc import ascent
import scipy.ndimage.interpolation as ndii


class TestTransformModes(ut.TestCase):
    def testModes(self):
        im = ascent()
        tvec = [200, 40]
        angle = 42

        for mode in ["constant", "reflect", "wrap"]:

            out1 = imreg.transform_img(im, tvec=tvec, angle=angle, mode=mode, bgval = 0.)
            out2 = utils._to_shape(ndii.shift(ndii.rotate(im, angle, order=1, mode=mode),
                                          tvec, mode=mode, order=1),
                               out1.shape)

            npt.assert_allclose(out1,out2)

    def testModesDict(self):
        im = ascent()
        tvec = [-67, 20]
        angle = 37
        scale = 1.

        for mode in ["constant", "reflect", "wrap"]:

            tdict = {"tvec":tvec,"angle":angle, "scale":scale}
            out1 = imreg.transform_img_dict(im, tdict, mode=mode, bgval = 0.)
            out2 = utils._to_shape(ndii.shift(ndii.rotate(im, angle, order=1, mode=mode),
                                          tvec, mode=mode, order=1),
                               out1.shape)

            npt.assert_allclose(out1,out2)


if __name__ == '__main__':
    ut.main()
