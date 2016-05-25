import unittest as ut

import imreg_dft.tform as tform


class TestImreg(ut.TestCase):
    def test_parse(self):
        instrs = [
            "scale: 1.8 +-8.99\n angle:186 \nshift (x, y): 35,44.2 success:1",
            "scale: 1.8 angle:186 \nshift (x, y): 35, 44.2 +-0.5 success:1",
        ]

        for instr in instrs:
            res = tform._str2tform(instr)
            self.assertAlmostEqual(res["scale"], 1.8)
            self.assertAlmostEqual(res["angle"], 186)
            self.assertAlmostEqual(res["tvec"][0], 44.2)  # y-component
            self.assertAlmostEqual(res["tvec"][1], 35)  # x-component


if __name__ == '__main__':
    ut.main()
