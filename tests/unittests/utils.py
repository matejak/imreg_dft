import unittest as ut

import numpy as np
import numpy.fft as fft
import numpy.testing

import imreg_dft.utils as utils


np.random.seed(108)


def _slice2arr(sli):
    res = []
    res.append(sli.start)
    res.append(sli.stop)
    res.append(res[1] - res[0])
    ret = np.array(res, int)
    return ret


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
            # TODO: unextend does not work 100% since it is not possible
            # principally
            # np.testing.assert_equal(what, undone)

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

    def testArgmax_ext(self):
        src = np.array([[1, 3, 1],
                        [0, 0, 0],
                        [1, 3.01, 0]])
        infres = utils.argmax_ext(src, 'inf')  # element 3.01
        self.assertEqual(tuple(infres), (2.0, 1.0))
        n10res = utils.argmax_ext(src, 10)  # element 1 in the rows with 3s
        n10res = np.round(n10res)
        self.assertEqual(tuple(n10res), (1, 1))

    def test_select(self):
        inshp = np.array((5, 8))

        start = np.array((0, 0))
        dim = np.array((2, 3))
        slis = utils.mkCut(inshp, dim, start)

        sliarrs = np.array([_slice2arr(sli) for sli in slis])
        np.testing.assert_array_equal(sliarrs[:, 2], dim)
        np.testing.assert_array_equal(sliarrs[:, 0], start)
        np.testing.assert_array_equal(sliarrs[:, 1], (2, 3))

        start = np.array((3, 6))
        dim = np.array((2, 3))
        slis = utils.mkCut(inshp, dim, start)

        sliarrs = np.array([_slice2arr(sli) for sli in slis])
        np.testing.assert_array_equal(sliarrs[:, 2], dim)
        np.testing.assert_array_equal(sliarrs[:, 0], (3, 5))
        np.testing.assert_array_equal(sliarrs[:, 1], inshp)

    def test_cuts(self):
        big = np.array((30, 50))
        small = np.array((20, 20))
        res = utils.getCuts(big, small, 0.25)
        # first is (0, 0), second is (0, 1)
        self.assertEquals(res[1][1], 5)
        # Last element of the row has beginning at 40
        self.assertEquals(res[5][1], 25)
        self.assertEquals(res[6][1], 30)
        self.assertEquals(res[7][1], 0)
        # (50 / 5) + 1 = 11th should be (5, 5) - 2nd of the 2nd row
        self.assertEquals(res[8], (5, 5))

        small = np.array((10, 20))
        res = utils.getCuts(big, small, 1.0)
        self.assertEquals(res[1], (0, 20))
        self.assertEquals(res[2], (0, 30))
        self.assertEquals(res[3], (10, 0))
        self.assertEquals(res[4], (10, 20))
        self.assertEquals(res[5], (10, 30))
        self.assertEquals(res[6], (20, 0))

    def test_cut(self):
        # Tests of those private functions are ugly
        res = utils._getCut(9, 3)
        np.testing.assert_array_equal(res, (0, 3, 6, 9))

        res = utils._getCut(80, 50)
        np.testing.assert_array_equal(res, (0, 50, 80))

    def test_decomps(self):
        smallshp = (30, 50)
        inarr = np.random.random(smallshp)
        recon = np.zeros_like(inarr)
        tileshp = (7, 6)
        decomps = utils.decompose(inarr, tileshp, 0.8)
        for decarr, start in decomps:
            sshp = decarr.shape
            recon[start[0]:start[0] + sshp[0],
                  start[1]:start[1] + sshp[1]] = decarr
        self.assertEqual(tileshp, decarr.shape)
        np.testing.assert_array_equal(inarr, recon)

    def test_fftshift(self):
        orig_arr = np.arange(12)
        shape = np.array((4, 3))
        orig_arr = orig_arr.reshape(shape)

        shifted_arr = np.fft.fftshift(orig_arr)

        for yy, shifted_row in enumerate(shifted_arr):
            for xx, val in enumerate(shifted_row):
                shifted_coord = np.array((yy, xx))
                # Of course this is equal
                self.assertEqual(val, shifted_arr[tuple(shifted_coord)])
                fixed_coord = utils._compensate_fftshift(shifted_coord, shape)
                # The actual test
                self.assertEqual(val, orig_arr[tuple(fixed_coord)])


class TestTiles(ut.TestCase):
    def testClusters(self):
        shifts = [(0, 1), (0, 1.1), (0.2, 1), (-0.1, 0.9), (-0.1, 0.8)]
        shifts = np.array(shifts)
        clusters = utils.get_clusters(shifts, 0.11)
        np.testing.assert_array_equal(clusters[0], (1, 1, 0, 1, 0))


if __name__ == '__main__':
    ut.main()
