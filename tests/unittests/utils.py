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

        yfreqs = fft.fftfreq(arr.shape[0])[:, np.newaxis]
        xfreqs = fft.fftfreq(arr.shape[1])[np.newaxis, :]
        weifun = xfreqs ** 2 + yfreqs ** 2

        ret = np.abs(dft) * weifun
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
            self.assertLess(dftscore, dftscore0 * 1.1)
            dftscore0 = dftscore

            undone = utils.unextend_by(ext, dst)
            self.assertEqual(what.shape, undone.shape)
            np.testing.assert_equal(what, undone)

    def _addFreq(self, src, vec):
        dom = np.zeros(src.shape)
        dom += np.arange(src.shape[0])[:, np.newaxis] * np.pi * vec[0]
        dom += np.arange(src.shape[1])[np.newaxis, :] * np.pi * vec[1]

        src += np.sin(dom)

        return src

    @staticmethod
    def _arrdiff(a, b):
        adiff = np.abs(a - b)
        ret = adiff.mean(), adiff.max()
        return ret

    def _wrapFilter(self, src, vecs, * args):
        dest = src.copy()
        for vec in vecs:
            self._addFreq(dest, vec)

        filtered = utils.imfilter(dest, * args)
        mold, mnew = [self._arrdiff(src, arr)[0] for arr in (dest, filtered)]
        self.assertGreater(mold * 1e-10, mnew)

    def testFilter(self):
        src = np.zeros((20, 30))

        self._wrapFilter(src, [(0.8, 0.8)], (0.8, 1.0))
        self._wrapFilter(src, [(0.1, 0.2)], None, (0.3, 0.4))

        src2 = self._addFreq(src.copy(), (0.1, 0.4))
        self._wrapFilter(src2, [(0.8, 0.8), (0.1, 0.2)], (0.8, 1.0), (0.3, 0.4))


if __name__ == '__main__':
    ut.main()
