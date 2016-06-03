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

        self.assertIn("one", wrapper)
        self.assertIn("1-two", wrapper)
        self.assertIn("5-three", wrapper)
        self.assertIn("1-four", wrapper)
        self.assertIn("five", wrapper)

        with self.assertRaises(AssertionError):
            reporting.ReportsWrapper(None)


if __name__ == '__main__':
    ut.main()
