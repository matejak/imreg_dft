#/bin/bash

test -n "$PYTHON" || PYTHON=python

die() { echo "$@" 1>&2 ; exit 1; }

VER=$1

test -n "$VER" || die "Supply the version number as the first argument"


# Check that we have it tagged
git tag | grep "^v$VER$" > /dev/null || die "The version is not tagged"

# Check that Python has the same version string
PYTHONPATH="../src" $PYTHON -c "import sys; from imreg_dft import __version__ as ver; sys.exit(0) if ver == '$VER' else sys.exit(1)" || die "The version of package doesn't match"

echo "All is OK for version '$VER'"
