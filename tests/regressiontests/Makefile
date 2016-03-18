# directory of regression tests(ie where this makefile is based in)
RTESTDIR := $(dir $(lastword $(MAKEFILE_LIST)))
PROOT = $(RTESTDIR)/../..
RSRCDIR = $(PROOT)/resources
EXAMPLESDIR = $(RSRCDIR)/examples

PYTHON ?= python
COVERAGE ?=
CMD =

NUL =

ifdef COVERAGE
# The shell script will see this and call ird using the coverage stuff as a wrapper
export CMD = $(COVERAGE) run -a --source imreg_dft $(PROOT)/src/imreg_dft/cli.py
endif

CHECKS = \
	check-translation \
	check-similarity \
	check-similarity2 \
	check-tiles \
	check-constraints1 \
	check-constraints2 \
	check-examples \
	$(NUL)

check: $(CHECKS)

clean:
	$(RM) .coverage
	$(RM) sample1.mat sample3c.mat

# Check that we can compute the translation and it is antisymetric
check-translation: $(EXAMPLESDIR)/sample1.png $(EXAMPLESDIR)/sample2.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) -19 79 0 1
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) 19 -79 0 1

# Check that we can compute also other things than translation
# and that it is also sortof antisymetric
check-similarity: $(EXAMPLESDIR)/sample1.png $(EXAMPLESDIR)/sample3.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) - - -30 1.25 --iter 2
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) - - 30 0.8 --iter 2

sample1.mat: $(EXAMPLESDIR)/sample1.png
	$(PYTHON) $(RTESTDIR)/img2mat.py $< $@ --var img --dummy-vars one,two

sample3c.mat: $(EXAMPLESDIR)/sample3c.jpg
	$(PYTHON) $(RTESTDIR)/img2mat.py $< $@ --var img

# Check that we can compute also other things than translation
# and on .mat files AND image files
check-similarity2: sample1.mat $(EXAMPLESDIR)/sample3.png $(EXAMPLESDIR)/sample1.png sample3c.mat
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) - - -30 1.25 --iter 2 --loader-opts in=img
	sh $(RTESTDIR)/check_similarity.sh $(word 4,$^) $(word 3,$^) - - 30 0.8 --iter 2

check-constraints1: $(EXAMPLESDIR)/constraints/tricky.png
	sh $(RTESTDIR)/check_similarity.sh $< $< - - 0 1
	sh $(RTESTDIR)/check_similarity.sh $< $< - - 180 1 --angle 140,20

check-constraints2: $(EXAMPLESDIR)/constraints/three.png $(EXAMPLESDIR)/constraints/two.png 
	sh $(RTESTDIR)/check_similarity.sh $^ - - 90 1
	sh $(RTESTDIR)/check_similarity.sh $^ - - 90 1 --tx 200,100

check-tiles: $(EXAMPLESDIR)/tiling/big.png $(EXAMPLESDIR)/tiling/small.png
	TOLER=8 sh $(RTESTDIR)/check_similarity.sh $^ 156 137 40.5 0.51 --tile

check-examples: $(RSRCDIR)/code/translation.py $(RSRCDIR)/code/similarity.py
	cd $(RSRCDIR)/code && IMSHOW=no $(PYTHON) $(word 1,$^) > /dev/null
	cd $(RSRCDIR)/code && IMSHOW=no $(PYTHON) $(word 2,$^) > /dev/null

.PHONY: $(CHECKS)
