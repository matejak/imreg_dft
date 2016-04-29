import unittest as ut

import imreg_dft.reporting as reporting


class TestReports(ut.TestCase):
    def testWrapper(self):
        wrapped = dict()
        wrapper = reporting.ReportsWrapper(wrapped)
        wrapper["one"] = 1
        wrapper.push_index(9)
        wrapper.pop_index(9)
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

        self.assertIn("one", wrapped)
        self.assertIn("1-two", wrapped)
        self.assertIn("5-three", wrapped)
        self.assertIn("1-four", wrapped)
        self.assertIn("five", wrapped)

        with self.assertRaises(AssertionError):
            wrapper_none = reporting.ReportsWrapper(None)


if __name__ == '__main__':
    ut.main()
