#/bin/bash

SUCCESS=yes

test -n "$PYTHON" || PYTHON=python

failure() { echo "$@" 1>&2 ; SUCCESS=no; }

VER=$1

test -n "$VER" || { echo "Supply the version number as the first argument" 1>&2; exit 1; }

TAGNAME="v$VER"
# Check that we have it tagged
git tag | grep "^$TAGNAME$" > /dev/null || failure "The version is not tagged (tag '$TAGNAME' missing)"

# Check that Python has the same version string
PYTHONPATH="../src" $PYTHON -c "import sys; from imreg_dft import __version__ as ver; sys.exit(0) if ver == '$VER' else sys.exit(1)" || failure "The version of package doesn't match"

test $SUCCESS == "yes" && { echo "All is OK for version '$VER'"; exit 0; }

echo "There were issues"
exit 1
