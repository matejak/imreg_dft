RSRCDIR = ../../resources/examples

check: check-translation check-similarity

# Check that we can compute the translation and it is antisymetric
check-translation: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample2.png
	sh check_similarity.sh $(word 1,$^) $(word 2,$^) -19 79 0 1
	sh check_similarity.sh $(word 2,$^) $(word 1,$^) 19 -79

# Check that we can compute also other things than translation
# and that it is also sortof antisymetric
check-similarity: $(RSRCDIR)/sample1.png $(RSRCDIR)/sample3.png
	sh check_similarity.sh $(word 1,$^) $(word 2,$^) 35 72 -30 1.25
	sh check_similarity.sh $(word 2,$^) $(word 1,$^) -54 -37 30 0.8
