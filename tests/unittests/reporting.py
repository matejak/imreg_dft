import unittest as ut

import imreg_dft.reporting as reporting


class TestReports(ut.TestCase):
    def testWrapper(self):
        wrapper = reporting.ReportsWrapper()
        wrapper["one"] = 1
        wrapper.push_prefix("9")
        wrapper.pop_prefix("9")
        wrapper.push_prefix("1-")
        wrapper["two"] = 2
        wrapper.push_prefix("5-")
        wrapper["three"] = 3

        with self.assertRaises(AssertionError):
            wrapper.pop_prefix("1-")
        wrapper.pop_prefix("5-")

        wrapper["four"] = 4
        wrapper.pop_prefix("1-")
        wrapper["five"] = 5

        self.assertIn("one", wrapper._stuff[""])
        self.assertIn("two", wrapper._stuff["1-"])
        self.assertIn("three", wrapper._stuff["5-"])
        self.assertIn("four", wrapper._stuff["1-"])
        self.assertIn("five", wrapper._stuff[""])


if __name__ == '__main__':
    ut.main()
