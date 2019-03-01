#!/bin/bash
echo ""
echo "#---------- $0 (Auspex) start..."
echo ""
#----- Simple script to capture NON-unittest preparation and invocation logic.

pwd
echo ""

# Don't forget to set the BBN_MEAS_FILE reference.
# Note:  QGL depends on git LFS data files -- without LFS checkout can
# also be used to test the dafa file signature reference failure logic
# (if eneriched)
export BBN_MEAS_FILE=test/test_measure.yml
echo "#-----Processing Auspex NON unit test modules (with BBN_MEAS_FILE=${BBN_MEAS_FILE})..."

# Aside -- plotting_test_xy currently fails

for moduleName in \
        plotting_test \
        plotting_test_arbitrary \
        plotting_test_avg \
        plotting_test_mesh \
        ; do
    echo ""
    echo "#..... Invoking ${moduleName} Auspex NON unit test module via ~:"
    export CMD="python test/${moduleName}.py"
    echo $CMD
    echo ""
    $CMD
    sleep 1

done


echo ""
echo "#---------- $0 (Auspex) stop."
echo ""
