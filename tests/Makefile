# directory of tests(ie where this makefile is based in)
TESTDIR := $(dir $(lastword $(MAKEFILE_LIST)))
COVERAGE ?=
PYTHON ?= python2

ifdef COVERAGE
RUNUNIT = $(COVERAGE) run --source=imreg_dft
else
RUNUNIT = $(PYTHON)
endif

help:
	@echo "Run either 'make check' or 'make check COVERAGE=<coverage command name>'"
	@echo "You may also append 'PYTHON=<python executable>' if you don't use coverage"

# if COVERAGE is defined, use it to run the first test
# and also to print a report at the end
#
# Unittests rewrite the coverage data and the regtests makefile
# runs them so they append to the coverage data
check:
ifdef COVERAGE
	$(RM) .coverage
endif
	$(RUNUNIT) $(TESTDIR)/unittests/__init__.py
	$(MAKE) -f regressiontests/Makefile check
	@echo '* * * * * * * * * * * * * * * * * * * * *'
	@echo ' Rejoice, tests have passed successfully!'
	@echo '* * * * * * * * * * * * * * * * * * * * *'
ifdef COVERAGE
	$(COVERAGE) report
endif
	$(MAKE) -f regressiontests/Makefile clean
