import unittest as ut

import numpy as np
import numpy.fft as fft
import numpy.testing

import imreg_dft.utils as utils


class TestUtils(ut.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.whatshape = (20, 11)
        self.whatsize = np.prod(self.whatshape)

    def testUndo(self):
        what = np.random.random(self.whatshape)
        wheres = [
            (20, 11),
            (21, 12),
            (22, 13),
            (50, 60),
        ]
        for whs in wheres:
            where = np.zeros(whs)
            embd = utils.embed_to(where, what.copy())
            undone = utils.undo_embed(embd, what.shape)
            self.assertEqual(what.shape, undone.shape, )
            np.testing.assert_equal(what, undone)

    def _dftscore(self, arr):
        # Measures the amount of DFT artifacts (in a quite strict manner)
        dft = fft.fft2(arr) * self.whatsize
        dft /= dft.size
        ret = np.log(np.abs(dft))
        return ret.sum()

    def testExtend(self):
        what = np.random.random((20, 11))
        whaty = what.shape[0]
        what[:] += np.arange(whaty, dtype=float)[:, np.newaxis] * 5 / whaty
        dftscore0 = self._dftscore(what)
        dsts = (2, 3, 4)
        for dst in dsts:
            ext = utils.extend_by(what, dst)

            # Bigger distance should mean better "DFT score"
            dftscore = self._dftscore(ext)
            self.assertLess(dftscore, dftscore0)
            dftscore0 = dftscore

            undone = utils.unextend_by(ext, dst)
            self.assertEqual(what.shape, undone.shape)
            np.testing.assert_equal(what, undone)


if __name__ == '__main__':
    ut.main()
