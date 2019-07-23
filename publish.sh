#!/bin/bash

# Usage: dist_pypi [production]
# Uses pypi test unless "production is supplied"

pkg_name=$(python setup.py --name)
version=$(python setup.py --version)
echo "DISTRIBUTING $pkg_name VERSION $version"

[[ $1 == "production" ]] && echo "*** PRODUCTION UPLOAD ***" || echo "*** TEST UPLOAD ***"
read -n 1 -s -r -p "Press any key to continue"

echo "** Removing dist directory **"
rm -rf dist

echo "** Creating source distribution **"
python setup.py sdist

echo "** Creating wheel distribution **"
python setup.py bdist_wheel

echo "** Uploading to test pypi **"
if [[ $1 == "production" ]]; then
    twine upload dist/*
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi

# Test with: pip install --extra-index-url https://test.pypi.org/simple/ bbndb

echo "** Creating conda skeleton **"
rm -rf skeleton
mkdir skeleton && pushd skeleton
if [[ $1 == "production" ]]; then
    conda skeleton pypi --version=$version $pkg_name
else
    conda skeleton pypi --version=$version --pypi-url https://test.pypi.io/pypi/ $pkg_name
fi
pushd $pkg_name

echo "** Please modify the meta.yaml in skeleton/$pkg_name to include \"noarch: python\" in the build section."
read -n 1 -s -r -p "Press any key to continue"

anaconda login
conda config --set anaconda_upload yes
conda build -c conda-forge .
conda config --set anaconda_upload no

popd && popd