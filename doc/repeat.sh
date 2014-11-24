#!/bin/sh

# default DIR (is where the images are)
test -z "$DIR" && DIR=../resources/examples

# We filter out --show from the actual command line
CMD="$(echo $@ | sed -e 's/ --show//' | sed -e 's/ --output [a-Z\._]*//')"

# Here we capture the output
OUT="$(cd $DIR && sh -c "$CMD")"

# What return code to tell to make
RC=$?

# Shell line imitation. $@ stands for the unfiltered command-line
echo "[user@linuxbox $(basename $DIR)]$ $@"
# If there is no output, don't echo anything to avoid the trailing newline
test -n "$OUT" && echo "$OUT"

# The previous statement would otherwise define the return code
exit $RC
