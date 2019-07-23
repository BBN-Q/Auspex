#!/bin/bash

pkg_name=$(python setup.py --name) #"bbndb"
version=$(python setup.py --version) # "2019.1.2"
echo "DISTRIBUTING $pkg_name VERSION $version"

echo "** Removing dist directory **"
rm -rf dist

echo "** Creating source distribution **"
python setup.py sdist

echo "** Creating wheel distribution **"
python setup.py bdist_wheel

echo "** Uploading to test pypi **"
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# For distribution:
# twine upload dist/*
# Test with:
# pip install --extra-index-url https://test.pypi.org/simple/ bbndb


# channels=$(conda config --show channels)

# echo "** Adding conda-forge to channels"
# conda config --add channels conda-forge

echo "** Creating conda skeleton **"
rm -rf skeleton
mkdir skeleton && pushd skeleton
conda skeleton pypi --version=$version --pypi-url https://test.pypi.io/pypi/ $pkg_name
pushd $pkg_name

echo "** Please modify the meta.yaml in skeleton/$pkg_name to include \"noarch: python\" in the build section."
read -n 1 -s -r -p "Press any key to continue"

anaconda login
conda config --set anaconda_upload yes
conda build -c conda-forge .
conda config --set anaconda_upload no

popd && popd