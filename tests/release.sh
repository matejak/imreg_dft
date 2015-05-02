#/bin/bash

PROOT="$(dirname $0)/.."
SUCCESS=yes

test -n "$PYTHON" || PYTHON=python

failure() { echo "$@" 1>&2 ; SUCCESS=no; }

VER=$1

test -n "$VER" || { echo "Supply the version number as the first argument" 1>&2; exit 1; }

TAGNAME="v$VER"
# Check that we have it tagged
git tag | grep "^$TAGNAME$" > /dev/null || failure "The version is not tagged (tag '$TAGNAME' missing)"

# Check that Python has the same version string
PYTHONPATH="$PROOT/src" $PYTHON -c "import sys; from imreg_dft import __version__ as ver; sys.exit(0) if ver == '$VER' else sys.exit(1)" || failure "The version of package doesn't match"

echo "Trying to generate documentation"
(cd "$PROOT/doc" && make clean > /dev/null && make html > /dev/null) || failure "Error(s) (re)generating HTML documentation, check that out"

echo "Trying to run tests"
(cd "$PROOT/tests" && make check > /dev/null) || failure "Error(s) running tests, check that out"

test $SUCCESS == "yes" && { echo "All is OK for version '$VER'"; exit 0; }

echo "There were issues"
exit 1

# Create an annotated tag
git tag -a $TAGNAME -m "Tagged version $VER"
# Set the right date
sed -i "s/TBA/$(date +%Y-%m-%d)/" $PROOD/doc/changelog.rst
# Upload sdist (sign) and upload documentation
(cd $PROOT; $PYTHON setup.py sdist upload -s; $PYTHON setup.py upload_docs)
