#!/bin/sh

die () {
	echo "$1"
       	exit 1
}

almost_equal () {
	python -c 'import sys; sys.exit(1 - ('"abs($1 - $2) < $3"'))'
	return $?
}

TX=$3
TY=$4
ANGLE=$5
SCALE=$6

test -z "$CMD" && CMD='ird'

TVEC=$($CMD "$1" "$2" --print-result --print-format '%(tx)d,%(ty)d,%(angle).3g,%(scale).3g' --iter 4)

GOTX=`echo $TVEC | cut -f 1 -d ,`
GOTY=`echo $TVEC | cut -f 2 -d ,`
GOTAng=`echo $TVEC | cut -f 3 -d ,`
GOTScale=`echo $TVEC | cut -f 4 -d ,`

test "$GOTX" -ne "$TX" -o "$GOTY" -ne "$TY" \
	&& die "Translation didn't work out, expected $TX,$TY got $GOTX,$GOTY"

test -n "$ANGLE" && \
	{ almost_equal "$GOTAng" "$ANGLE" 0.1 \
		|| die "Angle didn't work out, expected $ANGLE got $GOTAng"; }

test -n "$SCALE" && \
	{ almost_equal "$GOTScale" "$SCALE" 0.05 \
		|| die "Scale didn't work out, expected $SCALE got $GOTScale"; }

exit 0
