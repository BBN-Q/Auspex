# libchannelizer

Provides wrappers around Intel IPP filtering functions for use in the Auspex channelizer filter.


# Building

## IPP Deployable Shared Library

Intel provides a
[custom_library_tool](https://software.intel.com/en-us/articles/intel-integrated-performance-primitives-intel-ipp-for-windows-deploying-applications-with-intel-ipp-dlls)
to create a small shared library with only the needed functions for deployment
purposes. We can then link `libchannelizer` against this and not have the full
IPP as an Auspex dependency. We use the `-build` option to just build the shared
library but you can use the `-save` option to see what it is doing.

### Linux

```shell
cryan $ cd /intel/compilers_and_libraries_2017.0.098/linux/ipp/tools/custom_library_tool
cryan $ ./ipp_custom_library_tool -build -n ippreduced -l "..../src/auspex/filters/libchannelizer/ipp_functions.txt" -o "..../src/auspex/filters/libchannelizer/" -c ~/intel/compilers_and_libraries
```

### Windows


## libchannelizer

We currently don't do any error checking so the compiler will warn about unused status variables.

### Linux

On Linux tell the linker to use a relative rpath so that when loading libchannelizer into Python it can find libippreduced

```shell
cryan $ source ~/intel/compilers_and_libraries/linux/ipp/bin/ippvars.sh intel64
cryan $ g++ -c -std=c++11 -fPIC -Wall -O3 channelizer.cpp
cryan $ g++ channelizer.o -shared -o libchannelizer.so -L. -lippreduced -Wl,-rpath=`$ORIGIN`
```

### Windows
