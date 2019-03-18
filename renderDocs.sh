#!/bin/bash
echo ""
echo "#---------- $0 (Auspex) start..."
echo ""
#----- Simple script to capture basic documentation rendering logic.

if [ ! -z $1 ]; then
    export TGT_TYPE=$1
    echo "TGT_TYPE: '${TGT_TYPE}' << via arg 1 [$1]"
else
    export TGT_TYPE=html
    echo "TGT_TYPE: '${TGT_TYPE}' (defaulted; NO input arguments)"
fi 

export szMakePath=`which make`
#echo "szMakePath: [${szMakePath}]"

export BUILDDIR=_build

echo ""
if [ -z ${szMakePath} ]; then
    echo "Skipping Sphinx Makefile target referencing << Make omitted << [${szMakePath}] << 'which make'"
    echo "...configuring for direct sphinx call."
    
    if [ ! ${TGT_TYPE} == "clean" ]; then

        #export SPHINXOPTS=
        export SPHINXBUILD=sphinx-build
        export PAPER=letter

        # Internal variables.
        #export PAPEROPT_a4= -D latex_paper_size=a4
        export PAPEROPT_letter="-D latex_paper_size=letter"
        #export ALLSPHINXOPTS=" -d ${BUILDDIR}/doctrees ${PAPEROPT_${PAPER}} ${SPHINXOPTS} ."
        export ALLSPHINXOPTS=" -d ${BUILDDIR}/doctrees ${PAPEROPT_letter} ${SPHINXOPTS} ."

        #export CMD="${SPHINXBUILD} -b html ${ALLSPHINXOPTS} ${BUILDDIR}/html"
        export CMD="${SPHINXBUILD} -b ${TGT_TYPE} ${ALLSPHINXOPTS} ${BUILDDIR}/${TGT_TYPE}"
    
    else
        export CMD="rm -rf ${BUILDDIR}/*"
    fi

else
    echo "Standard (make managed) Sphinx call << Make exists << [${szMakePath}] << 'which make'"
    export CMD="make ${TGT_TYPE}"
fi

cd doc

echo ""
echo "#-----Building Auspex docs under:"
pwd

echo ""
echo ${CMD}
echo ""
${CMD}
echo ""


echo "Target Auspex documents list as follows:"

# Show it if target NOT clean 
echo ""
if [ ! ${TGT_TYPE} == "clean" ]; then
    cd ${BUILDDIR}/${TGT_TYPE}
else
    cd ${BUILDDIR}
fi

pwd
ls -al


echo ""
echo "#---------- $0 (Auspex) stop."
echo ""

