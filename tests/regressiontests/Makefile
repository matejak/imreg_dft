# directory of regression tests(ie where this makefile is based in)
RTESTDIR := $(dir $(lastword $(MAKEFILE_LIST)))
RSRCDIR = $(RTESTDIR)../../resources/examples
COVERAGE =
COVER_PREFIX =

ifdef COVERAGE
# The shell script will see this and call ird using the coverage stuff as a wrapper
export COVER_PREFIX = $(COVERAGE) run -a --source imreg_dft
endif

check: check-translation check-similarity

clean:
	$(RM) .coverage

# Check that we can compute the translation and it is antisymetric
check-translation: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample2.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) -19 79 0 1
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) 19 -79

# Check that we can compute also other things than translation
# and that it is also sortof antisymetric
check-similarity: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample3.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) 35 72 -30 1.25
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) -54 -37 30 0.8

.PHONY: check clean check-translation check-similarity
