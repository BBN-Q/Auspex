#!/bin/bash
echo ""
echo "#---------- $0 (Auspex) start..."
echo ""
#----- Simple script to capture unittest preparation and invocation logic.

pwd
echo ""

# Don't forget to set the BBN_MEAS_FILE reference.
# Note:  QGL depends on git LFS data files -- without LFS checkout can
# also be used to test the dafa file signature reference failure logic
# (if eneriched)
export BBN_MEAS_FILE=test/test_measure.yml
echo "#-----Unit-Testing Auspex (BBN_MEAS_FILE=${BBN_MEAS_FILE}) via ~:"

# Use -v option: verbose mode
# Use -f option: fail-fast mode
export CMD="python -m unittest discover . -v -f"
#export CMD="python -m unittest discover . -v"

# Individual tests stubbed out out here for piece-meal testing.
#export CMD="python -m unittest test/test_adapt_1d.py -v -f"
#export CMD="python -m unittest test/test_average.py -v -f"
#export CMD="python -m unittest test/test_bitfields.py -v -f"
#export CMD="python -m unittest test/test_buffer.py -v -f"
#export CMD="python -m unittest test/test_correlator.py -v -f"
#export CMD="python -m unittest test/test_experiment.py -v -f"
#export CMD="python -m unittest test/test_instrument.py -v -f"
#export CMD="python -m unittest test/test_pulsecal.py -v -f"
#export CMD="python -m unittest test/test_qubitexpfactory.py -v -f"
#export CMD="python -m unittest test/test_singleshot_filter.py -v -f"
#export CMD="python -m unittest test/test_sweeps.py -v -f"
#export CMD="python -m unittest test/test_write.py -v -f"

echo $CMD
echo ""
$CMD


echo ""
echo "#---------- $0 (Auspex) stop."
echo ""
