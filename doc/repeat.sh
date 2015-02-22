#!/bin/sh

# What env vars we accept:
# DIR: The execution directory (+ the last component will be displayed at the shell prompt)
# OUT: The output to display - don't execute any command if it is supplied
# PRIV: Private addition to the command - do, but don't display!

# default DIR (is where the images are)
test -z "$DIR" && DIR=../resources/examples

# We filter out --show from the actual command line
CMD="$(echo $@ | sed -e 's/ --show//' | sed -e 's/ --output [a-Z\._]*//')"

# Here we capture the output
# We may override the output by passing it as env var
test -n "$OUT" || OUT="$(cd $DIR && sh -c "$CMD $PRIV")"

# What return code to tell to make
RC=$?

# Shell line imitation. $@ stands for the unfiltered command-line
echo "[user@linuxbox $(basename $(cd $DIR && pwd))]$ $@"

# We transform /home/myname/somewhere/imreg_dft to /home/user/imreg_dft
# just to be consistent with the rest of the docs :-)
OUT="$(echo -e "$OUT" | sed -e "s/$(whoami)\/.*imreg_/user\/imreg_/g")"

# If there is no output, don't echo anything to avoid the trailing newline
test -n "$OUT" && echo "$OUT"

# The previous statement would otherwise define the return code
exit $RC
