#!/bin/sh

die () {
	echo "$1"
       	exit 1
}

almost_equal () {
	python -c 'import sys; sys.exit(1 - ('"abs($1 - $2) < $3"'))'
	return $?
}

TPL="$1"
shift
IMG="$1"
shift

TX=$1
shift
TY=$1
shift
ANGLE=$1
shift
SCALE=$1
shift

# Now $@ refers to the rest we want to pass to ird

test -z "$CMD" && CMD='ird'

TVEC=$($CMD "$TPL" "$IMG" --print-result --print-format '%(tx)d,%(ty)d,%(angle).8g,%(scale).8g' $@)

test $? -eq 0 || die "ird terminated with an error"

GOTX=`echo $TVEC | cut -f 1 -d ,`
GOTY=`echo $TVEC | cut -f 2 -d ,`
GOTAng=`echo $TVEC | cut -f 3 -d ,`
GOTScale=`echo $TVEC | cut -f 4 -d ,`

test "$GOTX" -ne "$TX" -o "$GOTY" -ne "$TY" \
	&& die "Translation didn't work out, expected $TX,$TY got $GOTX,$GOTY"

test -z "$DANGLE" && DANGLE=0.2
test -n "$ANGLE" && \
	{ almost_equal "$GOTAng" "$ANGLE" "$DANGLE" \
		|| die "Angle didn't work out, expected $ANGLE got $GOTAng"; }

test -z "$DSCALE" && DSCALE=0.05
test -n "$SCALE" && \
	{ almost_equal "$GOTScale" "$SCALE" "$DSCALE" \
		|| die "Scale didn't work out, expected $SCALE got $GOTScale"; }

exit 0
