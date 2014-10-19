import sys
import os.path
import unittest as ut


tl = ut.defaultTestLoader
tests = tl.discover(os.path.dirname(os.path.abspath(__file__)),
                    r'[a-z]*.py')
suite = ut.TestSuite()

# We are the last discovered test
for test in tests:
    suite.addTest(test)

runner = ut.TextTestRunner(verbosity=2)
ret = runner.run(suite)

retcode = 1
if ret.wasSuccessful():
    retcode = 0
sys.exit(retcode)
