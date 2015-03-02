#!/bin/sh

die () {
	echo "$1"
       	exit 1
}


# + 0.0001 <= handling of tricky corner cases
almost_equal () {
	python -c 'import sys; sys.exit(1 - ('"abs($1 - $2) <= $3 * $TOLER + 0.0001"'))'
	return $?
}

almost_equal_angle () {
	python -c 'import sys; sys.exit(1 - ('"abs(($1 - $2 + 180) % 360 - 180) <= $3 * $TOLER + 0.0001"'))'
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
test -z "$TOLER" && TOLER='1'

# Precision of value and error has to be the same! There are some corner cases (e.g. angle = 90)
TVEC=$($CMD "$TPL" "$IMG" --print-result --print-format '%(tx).6g,%(ty).6g,%(angle).6g,%(scale).6g,%(Dangle).6g,%(Dscale).6g' $@)

test $? -eq 0 || die "ird terminated with an error"

GOTX=`echo $TVEC | cut -f 1 -d ,`
GOTY=`echo $TVEC | cut -f 2 -d ,`
GOTAng=`echo $TVEC | cut -f 3 -d ,`
GOTScale=`echo $TVEC | cut -f 4 -d ,`
DANGLE=`echo $TVEC | cut -f 5 -d ,`
DSCALE=`echo $TVEC | cut -f 6 -d ,`


# x$... because $... may be '-' and the test command may understand it its own way
test "x$TX" != 'x-' && \
	{ almost_equal "$GOTX" "$TX" 0.5 \
		|| die "X translation didn't work out, expected $TX, got $GOTX"; }

test "x$TY" != 'x-' && \
	{ almost_equal "$GOTY" "$TY" 0.5 \
		|| die "Y translation didn't work out, expected $TY, got $GOTY"; }

test -n "$ANGLE" -a "x$ANGLE" != 'x-' && \
	{ almost_equal_angle "$GOTAng" "$ANGLE" "$DANGLE" \
		|| die "Angle didn't work out, expected $ANGLE got $GOTAng"; }

test -n "$SCALE" -a "x$SCALE" != 'x-' && \
	{ almost_equal "$GOTScale" "$SCALE" "$DSCALE" \
		|| die "Scale didn't work out, expected $SCALE got $GOTScale"; }

exit 0
