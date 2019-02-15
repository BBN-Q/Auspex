# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# This file is originally from PyQLab (http://github.com/bbn-q/PyQLab)

import os, os.path
import sys
from shutil import move
from io import StringIO
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml

# Use when wanting to generate fake data
# or to avoid loading libraries that may
# interfere with desired operation. (e.g.
# when scraping modules in Auspex)
auspex_dummy_mode = False

# If this is True, then close the last
# plotter before starting a new one.
single_plotter_mode = True

# This holds a reference to the most
# recent plotters.
last_plotter_process = None
last_extra_plotter_process = None

# Config directory
meas_file         = None
AWGDir            = None
ConfigurationFile = None
KernelDir         = None
LogDir            = None

# ----- No Holzworth warning Start...
# Added the followiing 25 Oct 2018 to test Instrument metaclass load introspection
# minimization (during import) which, with holzworth.py module deltas in-turn,
# bars holzworth warnings
#
# Just set the initial defaults
# (rather than fancy try/(catch) except detection logic):
tgtInstrumentClass      = None
tgtFilterClass          = None
bEchoInstrumentMetaInit = False
# unlike dummy_mode usage altered code attempts to invoke module loads and
# exercise error processing; set this true, however, to enable stepping beyond
# and/or simulating the unavailable library use cases.  Leaving it false
# (the default case) paints the errors and warns the user as ecpected.
#
# wants to be true to support discovery unit tests with original default behavior
bUseMockOnLoadError     = True
# ----- No Holzworth warning Stop.

def find_meas_file():
    global meas_file
    # First default to any manually set options in the globals
    if meas_file:
        return os.path.abspath(meas_file)
    # Next use the meas file location in the environment variables
    if os.getenv('BBN_MEAS_FILE'):
        return os.getenv('BBN_MEAS_FILE')
    raise Exception("Could not find the measurement file in the environment variables or the auspex globals.")

class Include():
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            self.data = yaml.load(f, Loader=yaml.RoundTripLoader)
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def items(self):
        return self.data.items()
    def keys(self):
        return self.data.keys()
    def write(self):
        with open(self.filename+".tmp", 'w') as fid:
            yaml.dump(self.data, fid, Dumper=yaml.RoundTripDumper)
        move(self.filename+".tmp", self.filename)
    def pop(self, key):
        return self.data.pop(key)

class Loader(yaml.RoundTripLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def include(self, node):
        shortname = self.construct_scalar(node)
        filename = os.path.abspath(os.path.join(
            self._root, shortname
        ))
        return Include(filename)

class Dumper(yaml.RoundTripDumper):
    def include(self, data):
        data.write()
        return self.represent_scalar(u'!include', data.filename)

class FlatDumper(yaml.RoundTripDumper):
    def include(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data.data)

def load_meas_file(filename=None):
    global LogDir, KernelDir, AWGDir, meas_file

    if filename:
        meas_file = filename
    else:
        meas_file = find_meas_file()

    with open(meas_file, 'r') as fid:
        Loader.add_constructor('!include', Loader.include)
        load = Loader(fid)
        code = load.get_single_data()
        load.dispose()

    # Get the config values out of the measure_file.
    if not 'config' in code.keys():
        raise KeyError("Could not find config section of the yaml file.")

    if 'AWGDir' in code['config'].keys():
        AWGDir = os.path.abspath(code['config']['AWGDir'])
    else:
        raise KeyError("Could not find AWGDir in the YAML config section")

    if 'KernelDir' in code['config'].keys():
        KernelDir = os.path.abspath(code['config']['KernelDir'])
    else:
        raise KeyError("Could not find KernelDir in the YAML config section")

    if 'LogDir' in code['config'].keys():
        LogDir = os.path.abspath(code['config']['LogDir'])
    else:
        raise KeyError("Could not find LogDir in the YAML config section")

    # Create directories if necessary
    for d in [KernelDir, LogDir]:
        if not os.path.isdir(d):
            os.mkdir(d)

    return code

def dump_meas_file(data, filename = "", flatten=False):
    d = Dumper if filename and not flatten else FlatDumper
    d.add_representer(Include, d.include)

    if filename:
        with open(filename+".tmp", 'w+') as fid:

            yaml.dump(data, fid, Dumper=d)
        # Upon success
        move(filename+".tmp", filename)
        with open(filename, 'r') as fid:
            contents = fid.read()
        return contents
    else:
        # dump to an IO stream:
        # note you need to use the FlatDumper for this to work
        out = StringIO()
        yaml.dump(data, out, Dumper=d)
        ret_string = out.getvalue()
        out.close()
        return ret_string

# Generalized MetaClass constraint processing, 30 Oct 2018 -- work-in-progress
#
from auspex.log import logger


__szSMI_LogTextPrefix="%s <currClName> '%s' %s.__init__(...)\n\r   << "

# TODO - should I rename this filterMetaInit >> true : skip it, false : keep it
#
def skipMetaInit (currClName, currClBases, currClDict, acceptClassRefz=None, bEchoDetails=False, szLogLabel="Meta?"):
    """
    Where acceptClassRefz is defined,
    determines equivalence/membership of currClName value with respect to
    the acceptClassRefz {string, list, set} object.
    Logic generalized here to support numerous MetaClass initialization
    constraint analysis/usage.
    """
    if None == currClName:
        raise Exception( "Bogus currClName [%s] cited!", currClName)

    if None == currClBases:
        raise Exception( "Bogus currClBases [%s] cited!", currClBases)

    if None == currClDict:
        raise Exception( "Bogus currClDict [%s] cited!", currClDict)

    bSkipMInit = False

    if not (None == acceptClassRefz):
        # acceptClassRefz defined
        #
        acceptedRefzType = type( acceptClassRefz)

        if str == acceptedRefzType:
            # Process acceptedRefzType as a single string value
            if not (currClName == acceptClassRefz):
                # No Match
                logger.debug( __szSMI_LogTextPrefix +
                    "!= '%s' <acceptClassRefz> %s\n\r",
                    "Skipping", currClName, szLogLabel, acceptClassRefz, acceptedRefzType)
                bSkipMInit = True
            else:
                # Matched
                if bEchoDetails:
                    logger.info( __szSMI_LogTextPrefix +
                        "== <acceptClassRefz> %s 8-)\n\r",
                        "Continuing", currClName, szLogLabel, acceptedRefzType)
        else:
            if list == acceptedRefzType or set == acceptedRefzType:
                # Process as a multi-entry list ['A', 'B'] or set {'A', 'B'}
                if not (currClName in acceptClassRefz):
                    # No match
                    logger.debug( __szSMI_LogTextPrefix +
                        "currClName is NOT an acceptClassRefz element:"  \
                        "\n\r      %s::%s\n\r",
                        "Skipping", currClName, szLogLabel, acceptedRefzType, acceptClassRefz)
                    bSkipMInit = True
                else:
                    # Matched
                    if bEchoDetails:
                        logger.info( __szSMI_LogTextPrefix +
                            "currClName is an element of acceptClassRefz:" \
                            "\n\r      %s::%s 8-)\n\r",
                            "Continuing", currClName, szLogLabel, acceptedRefzType, acceptClassRefz)
            #
            else:
                raise Exception( "Unhandled acceptedRefzType: {} cited!".format( acceptedRefzType))

    # else acceptClassRefz NOT defined; default behavior (no skip) applies

    if bEchoDetails and not bSkipMInit:
        # Optionally paint the Instrument metaclass _init_ Parameters
        logger.info( "%s %s.__init__( currClName, currClBases, currClDict):" \
           "\n\r   --  currClName: %s" \
           "\n\r   -- currClBases: %s" \
           "\n\r   --  currClDict: %s\n\r",
           "++", szLogLabel, currClName, currClBases, currClDict)

    return bSkipMInit

#----- end skipMetaInit function definition.


#--- generalize logic that looks for a lower case sub-set reference against
# a given str, list, set acceptClassRefz

def disjointNameRefz (tgtBaseClName, acceptClassRefz=None, bEchoDetails=False, szLogLabel="DisJ?"):
    """
    Where acceptClassRefz is defined,
    determines whether the tgtBaseClName is disjoint from the one or more
    acceptClassRefz {string, list, set} object value[s].  More precisely,
    determines if (lower-case) tgtBaseClName is a substring of the
    (lowercase) acceptClassRefz elements.
    Logic generalized here to support numerous MetaClass initialization
    constraint analysis/usage.
    """
    if None == tgtBaseClName:
        raise Exception( "Bogus tgtBaseClName [%s] cited!", tgtBaseClName)

    bDisjointForRefz = False

    szTgtSubKey = tgtBaseClName.lower()

    if not (None == acceptClassRefz):
        # acceptClassRefz defined
        #
        acceptedRefzType = type( acceptClassRefz)

        szSE_MsgMask = "%s %s." \
            "\n\r   << <szTgtSubKey> '%s' %s '%s' <acceptClassRefz> [lc] substring."

        szME_MsgMask = "%u/%u: %s << <szTgtSubKey> '%s' %s '%s' <currAcceptClName> %s element substring.\n\r"

        if str == acceptedRefzType:
            # Process acceptedRefzType as a single string value
            # One answer to deal with; yes or no.
            if -1 == acceptClassRefz.lower().find( szTgtSubKey):
                # No match
                logger.debug( (szSE_MsgMask + "\n\r"), "Skipping",
                    szLogLabel, szTgtSubKey, "NOT", acceptClassRefz)
                bDisjointForRefz = True
            else:
                # Matched
                if bEchoDetails:
                    logger.info( szSE_MsgMask, "Continuing",
                        szLogLabel, szTgtSubKey, "noted as", acceptClassRefz)
                # bDisjointForRefz remains false

        else:
            if list == acceptedRefzType or set == acceptedRefzType:
                # Process as a multi-entry list ['A', 'B'] or set {'A', 'B'}
                # Multiple possibilities; return false on first match
                nIndex = 1;
                nCount = len( acceptClassRefz)
                bMatched = False
                for currAcceptClName in acceptClassRefz:
                    if -1 == currAcceptClName.lower().find( szTgtSubKey):
                        # No match
                        logger.debug( szME_MsgMask,
                            nIndex, nCount, "NoMatch", szTgtSubKey, "NOT", currAcceptClName, acceptedRefzType)
                    else:
                        bMatched = True
                        if bEchoDetails:
                            logger.info( szME_MsgMask,
                                nIndex, nCount, "Matched", szTgtSubKey, "noted as", currAcceptClName, acceptedRefzType)
                        break

                    nIndex += 1

                # end for all acceptClassRefz elements

                bDisjointForRefz = not bMatched

                if bDisjointForRefz:
                    logger.debug( "Skipping %s;  %s << bDisjointForRefz\n\r",
                        szLogLabel, bDisjointForRefz)
                else:
                    if bEchoDetails:
                        logger.info( "Continuing %s;  %s << bDisjointForRefz",
                            szLogLabel, bDisjointForRefz)
                #
            else:
                raise Exception( "Unhandled acceptedRefzType: {} cited!".format( acceptedRefzType))

    # else acceptClassRefz NOT defined; default behavior; disjointness not
    # determinte -- no skip applies

    return bDisjointForRefz

#----- end disjointNameRefz function definition.
