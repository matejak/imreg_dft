# directory of regression tests(ie where this makefile is based in)
RTESTDIR := $(dir $(lastword $(MAKEFILE_LIST)))
PROOT = $(RTESTDIR)/../..
RSRCDIR = $(PROOT)/resources/examples

PYTHON ?= python
COVERAGE ?=
CMD =

NUL =

ifdef COVERAGE
# The shell script will see this and call ird using the coverage stuff as a wrapper
export CMD = $(COVERAGE) run -a --source imreg_dft $(PROOT)/src/imreg_dft/cli.py
endif

check: \
	check-cells \
	check-translation \
	check-similarity \
	check-similarity2 \
	check-constraints1 \
	check-constraints2 \
	$(NUL)

clean:
	$(RM) .coverage
	$(RM) sample1.mat sample3c.mat

# Check that we can compute the translation and it is antisymetric
check-translation: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample2.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) -19 79 0 1
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) 19 -79 0 1

# Check that we can compute also other things than translation
# and that it is also sortof antisymetric
check-similarity: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample3.png
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) 35 73 -30 1.25 --iter 2
	sh $(RTESTDIR)/check_similarity.sh $(word 2,$^) $(word 1,$^) -54 -37 30 0.8 --iter 2

sample1.mat: $(RSRCDIR)/sample1.png
	$(PYTHON) $(RTESTDIR)/img2mat.py $< $@ --var img --dummy-vars one,two

sample3c.mat: $(RSRCDIR)/sample3c.jpg
	$(PYTHON) $(RTESTDIR)/img2mat.py $< $@ --var img

# Check that we can compute also other things than translation
# and on .mat files AND image files
check-similarity2: sample1.mat $(RSRCDIR)/sample3.png $(RSRCDIR)/sample1.png sample3c.mat
	sh $(RTESTDIR)/check_similarity.sh $(word 1,$^) $(word 2,$^) 35 73 -30 1.25 --iter 2 --loader-opts in=img
	sh $(RTESTDIR)/check_similarity.sh $(word 4,$^) $(word 3,$^) -54 -37 30 0.8 --iter 2

check-constraints1: $(RSRCDIR)/constraints/tricky.png
	sh $(RTESTDIR)/check_similarity.sh $< $< - - 0 1
	sh $(RTESTDIR)/check_similarity.sh $< $< - - 180 1 --angle 140,20

check-constraints2: $(RSRCDIR)/constraints/three.png $(RSRCDIR)/constraints/two.png 
	sh $(RTESTDIR)/check_similarity.sh $^ 0 0 90 1
	sh $(RTESTDIR)/check_similarity.sh $^ 179 0 90 1 --tx 200,100

check-cells: $(RSRCDIR)/cells/big-fluo.jpg $(RSRCDIR)/cells/small-dhm.jpg
	TOLER=4 sh $(RTESTDIR)/check_similarity.sh $^ 32 26 0.3 1 --tile --filter-pcorr 4 --lowpass 0.6,0.7 --angle 0,20 --scale 1,0.2

.PHONY: \
	check \
	clean \
	check-translation \
	check-similarity \
	check-cells \
	$(NUL)
