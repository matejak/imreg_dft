import unittest as ut

import numpy as np
import numpy.testing

import imreg_dft.utils as utils


class TestEmbed(ut.TestCase):
    def testUndo(self):
        what = np.random.random((20, 11))
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


if __name__ == '__main__':
    ut.main()
